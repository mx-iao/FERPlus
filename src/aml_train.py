#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging

from models import *
from ferplus import *

import cntk as C

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}

# List of folders for training, validation and test.
train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid'] 
test_folders  = ['FER2013Test']

def cost_func(training_mode, prediction, target):
    '''
    We use cross entropy in most mode, except for the multi-label mode, which require treating
    multiple labels exactly the same.
    '''
    train_loss = None
    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # Cross Entropy.
        train_loss = C.negate(C.reduce_sum(C.element_times(target, C.log(prediction)), axis=-1))
    elif training_mode == 'multi_target':
        train_loss = C.negate(C.log(C.reduce_max(C.element_times(target, prediction), axis=-1)))

    return train_loss
    
def main(base_folder, output_dir, training_mode='majority', learning_rate=0.05, momentum_rate=0.9, l2_reg_weight=0.0, minibatch_size=64,
    model_name='VGG13', max_epochs=100):

    os.makedirs(output_dir, exist_ok=True)

    # creating logging file 
    logging.basicConfig(filename = os.path.join(output_dir, "train.log"), filemode = 'w', level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Starting with training mode {} using {} model and max epochs {}.".format(training_mode, model_name, max_epochs))

    # create the model
    num_classes = len(emotion_table)
    model       = build_model(num_classes, model_name)

    # set the input variables.
    input_var = C.input_variable((1, model.input_height, model.input_width), np.float32)
    label_var = C.input_variable((num_classes), np.float32)
    
    # read FER+ dataset.
    logging.info("Loading data...")
    train_params        = FERPlusParameters(num_classes, model.input_height, model.input_width, training_mode, False)
    test_and_val_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", True)

    train_data_reader   = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader     = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader    = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)
    
    # print summary of the data.
    display_summary(train_data_reader, val_data_reader, test_data_reader)
    
    # get the probalistic output of the model.
    z    = model.model(input_var)
    pred = C.softmax(z)
    
    epoch_size     = train_data_reader.size()

    # Training config
    lr_per_minibatch       = [learning_rate]*20 + [learning_rate / 2.0]*20 + [learning_rate / 10.0]
    lr_schedule            = C.learning_rate_schedule(lr_per_minibatch, unit=C.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule = C.momentum_schedule(momentum_rate, minibatch_size=minibatch_size)

    # loss and error cost
    train_loss = cost_func(training_mode, pred, label_var)
    pe         = C.classification_error(z, label_var)

    # construct the trainer
    learner = C.momentum_sgd(z.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)

    progress_printer = C.logging.ProgressPrinter(freq=50, tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (train_loss, pe), learner, progress_printer)
    #trainer = C.Trainer(z, (train_loss, pe), learner)

    # Get minibatches of images to train with and perform model training
    max_val_accuracy    = 0.0
    final_test_accuracy = 0.0
    best_test_accuracy  = 0.0

    logging.info("Start training...")
    epoch      = 0
    best_epoch = 0
    while epoch < max_epochs: 
        train_data_reader.reset()
        val_data_reader.reset()
        test_data_reader.reset()
        
        # Training 
        start_time = time.time()
        training_loss = 0
        training_accuracy = 0
        while train_data_reader.has_more():
            images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({input_var : images, label_var : labels})

            # keep track of statistics.
            training_loss     += trainer.previous_minibatch_loss_average * current_batch_size
            training_accuracy += trainer.previous_minibatch_evaluation_average * current_batch_size
                
        training_accuracy /= train_data_reader.size()
        training_accuracy = 1.0 - training_accuracy

        trainer.summarize_training_progress()
        
        # Validation
        val_accuracy = 0
        while val_data_reader.has_more():
            images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
            val_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size
            
        val_accuracy /= val_data_reader.size()
        val_accuracy = 1.0 - val_accuracy

        trainer.summarize_test_progress()
        
        # if validation accuracy goes higher, we compute test accuracy
        test_run = False
        if val_accuracy > max_val_accuracy:
            best_epoch = epoch
            max_val_accuracy = val_accuracy

            trainer.save_checkpoint(os.path.join(output_dir, "model_{}".format(best_epoch)))

            test_run = True
            test_accuracy = 0
            while test_data_reader.has_more():
                images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
                test_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size

            trainer.summarize_test_progress()
            
            test_accuracy /= test_data_reader.size()
            test_accuracy = 1.0 - test_accuracy
            final_test_accuracy = test_accuracy
            if final_test_accuracy > best_test_accuracy: 
                best_test_accuracy = final_test_accuracy

        logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
        logging.info("  training loss:\t{:e}".format(training_loss))
        logging.info("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
        logging.info("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
        if test_run:
            logging.info("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
            
        epoch += 1

    logging.info("")
    logging.info("Best validation accuracy:\t\t{:.2f} %, epoch {}".format(max_val_accuracy * 100, best_epoch))
    logging.info("Test accuracy corresponding to best validation:\t\t{:.2f} %".format(final_test_accuracy * 100))
    logging.info("Best test accuracy:\t\t{:.2f} %".format(best_test_accuracy * 100))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", 
                        "--base_folder", 
                        type = str, 
                        help = "Base folder containing the training, validation and testing data.", 
                        required = True)
    parser.add_argument("--output_dir", 
                        type = str, 
                        help = "Output directory to write model checkpoints to", 
                        required = True)
    parser.add_argument("-m", 
                        "--training_mode", 
                        type = str,
                        default='majority',
                        help = "Specify the training mode: majority, probability, crossentropy or multi_target.")
    parser.add_argument("--learning_rate",
                        type = float,
                        default = "0.05",
                        required = False,
                        help = "Learning rate")
    parser.add_argument("--momentum_rate",
                        type = float,
                        default = "0.9",
                        required = False,
                        help = "Momentum value for the momentum schedule")
    parser.add_argument("--l2_reg_weight",
                        type = float,
                        default = "0.0",
                        required = False,
                        help = "L2 regularization weight per sample for learner")
    parser.add_argument("--minibatch_size",
                        type = int,
                        default = "64",
                        required = False,
                        help = "Minibatch size")

    # original code, minibatch_size=32, but for our purposes we want to just expedite training time right now

    args = parser.parse_args()
    main(args.base_folder, args.output_dir, args.training_mode, args.learning_rate, args.momentum_rate, args.l2_reg_weight, args.minibatch_size)