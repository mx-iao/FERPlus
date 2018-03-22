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
from distributed_ferplus import *

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
    
def main(base_folder, output_dir, training_mode='majority', learning_rate=0.05, momentum_rate=0.9, l2_reg_weight=0.0, 
    model_name='VGG13', max_epochs = 100):

    # create the model
    num_classes = len(emotion_table)
    model       = build_model(num_classes, model_name)

    # set the input variables.
    input_var = C.input_variable((1, model.input_height, model.input_width), np.float32)
    label_var = C.input_variable((num_classes), np.float32)
    
    # read FER+ dataset.
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
    minibatch_size = 32

    # Training config
    lr_per_minibatch       = [learning_rate]*20 + [learning_rate / 2.0]*20 + [learning_rate / 10.0]
    lr_schedule            = C.learning_rate_schedule(lr_per_minibatch, unit=C.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule = C.momentum_schedule(momentum_rate, minibatch_size=minibatch_size)

    # loss and error cost
    train_loss = cost_func(training_mode, pred, label_var)
    pe         = C.classification_error(z, label_var)

    # construct the trainer
    learner = C.momentum_sgd(z.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)

    # Construct the distributed learner
    distributed_learner = C.train.distributed.data_parallel_distributed_learner(learner)

    num_partitions = C.train.distributed.Communicator.num_workers()
    partition = C.train.distributed.Communicator.rank()

    progress_printer = C.logging.ProgressPrinter(freq=50, tag='Training', rank=partition, num_epochs=max_epochs)
    trainer = C.Trainer(z, (train_loss, pe), distributed_learner, progress_printer)

    # Get minibatches of images to train with and perform model training
    max_val_accuracy    = 0.0
    final_test_accuracy = 0.0
    best_test_accuracy  = 0.0

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
            images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size, num_data_partitions=num_partitions, partition_index=partition)

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
            images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size, num_data_partitions=num_partitions, partition_index=partition)
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
                images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size, num_data_partitions=num_partitions, partition_index=partition)
                test_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size

            trainer.summarize_test_progress()
            
            test_accuracy /= test_data_reader.size()
            test_accuracy = 1.0 - test_accuracy
            final_test_accuracy = test_accuracy
            if final_test_accuracy > best_test_accuracy: 
                best_test_accuracy = final_test_accuracy
            
        epoch += 1

    # Output the best checkpointed model to ONNX format, only save on master process
    if C.train.distributed.Communicator.is_main():   
        best_model = C.Function.load(os.path.join(output_dir, "model_{}".format(best_epoch)))
        inference_model = C.as_composite(best_model.outputs[0].owner)
        #or possibly: 
        #inference_model = C.as_composite(best_model[0].owner)
        print(inference_model)
        inference_model.save(os.path.join(output_dir, "model.onnx"), format=C.ModelFormat.ONNX)
    
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



    args = parser.parse_args()
    main(args.base_folder, args.output_dir, args.training_mode, args.learning_rate, args.momentum_rate, args.l2_reg_weight)

    C.train.distributed.Communicator.finalize()