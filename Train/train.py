import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import numpy as np
import pandas as pd
from random import shuffle
import sys
from tensorflow.python.keras.callbacks import Callback, LearningRateScheduler
from MSF_long_term import *
from MSF_short_term import *
import argparse
from evaluate import *

parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, default='./')
parser.add_argument('--read_sig_path', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--model_save_dir', type=str, default='./')
parser.add_argument('--model_name', type=str, default='bsMSF')

parser.add_argument('--lr', type=float, default=0.00005)

parser.add_argument('--input_length', type=int, default=2500)
parser.add_argument('--dilation', type=int, default=2)
parser.add_argument('--model_width', type=int, default=62)
parser.add_argument('--num_channel', type=int, default=1)

parser.add_argument('--output_nums', type=int, default=5)

parser.add_argument('--multi_path', type=str, default='long_term')
parser.add_argument('--forecasting_module', type=str, default=True)

args, unknown = parser.parse_known_args()
if __name__ == '__main':
    print(args, file=sys.stdout, flush=True)

    if args.multi_path == 'long_term':
        model = MSFMA_long(args.input_length, args.num_channel, args.model_width, args.dilation, args.output_nums, args.forecasting_module).HGSPU()
        
    else:
        model = MSFMA_short(args.input_length, args.num_channel, args.model_width, args.dilation, args.output_nums, args.forecasting_module).HGSPU()
    
    for i in range(5):
        # load dataset(read_path, reap_sig_path)
        
        print('CV = ', i)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit([_X_train_cv1_inp1[:,:1250],_X_train_cv1_inp1[:,1250:], _X_train_cv1_inp2[:,:1250], _X_train_cv1_inp2[:,1250:], _X_train_cv1_inp3[:,:1250], _X_train_cv1_inp3[:,1250:]], y_train_onehot, epochs = 20, batch_size = 16, validation_data=([_X_test_cv1_inp1[:,:1250], _X_test_cv1_inp1[:,1250:], _X_test_cv1_inp2[:,:1250], _X_test_cv1_inp2[:,1250:], _X_test_cv1_inp3[:,:1250], _X_test_cv1_inp3[:,1250:]], y_test_onehot), callbacks = [callback])
        
        model.save(args.model_save_dir + modelName + '_' + str(i) + '_' + 'pred_5hr' + '.h5')


        result = model.predict([_X_test_cv1_inp1[:,:1250], _X_test_cv1_inp1[:,1250:], _X_test_cv1_inp2[:,:1250], _X_test_cv1_inp2[:,1250:], _X_test_cv1_inp3[:,:1250], _X_test_cv1_inp3[:,1250:]])

        eval_pred(y_test_onehot, result,  i, args.output_nums, args.model_save_dir, args.model_name) 
