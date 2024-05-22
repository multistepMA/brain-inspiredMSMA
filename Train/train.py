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


parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, default='./')              # Read labeling data 
parser.add_argument('--read_sig_path', type=str, default='./')          # Read signal data
parser.add_argument('--batch_size', type=int, default=16)               # Batch size
parser.add_argument('--epochs', type=int, default=20)                   # Ephochs
parser.add_argument('--model_save_dir', type=str, default='./')         # Path of saving model
parser.add_argument('--model_name', type=str, default='bsMSF')          # Model name saved

parser.add_argument('--lr', type=float, default=0.00005)                # Learning rate

parser.add_argument('--input_length', type=int, default=7500)           # Input length(Sampling rate * 30 s)
parser.add_argument('--dilation', type=int, default=2)                  # Causal convolution filter size
parser.add_argument('--model_width', type=int, default=62)              # Model width
parser.add_argument('--num_channel', type=int, default=1)               # Number of ECG channel

parser.add_argument('--output_nums', type=int, default=5)               # Number of multi-step 

parser.add_argument('--multi_path', type=str, default='long_term')      # Path selection
parser.add_argument('--forecasting_module', type=str, default=True)     # Forecasting module selection

args, unknown = parser.parse_known_args()
if __name__ == '__main':
    print(args, file=sys.stdout, flush=True)

    if args.multi_path == 'long_term':
        model = MSFMA_long(args.input_length, args.num_channel, args.model_width, args.dilation, args.output_nums, args.forecasting_module).HGSPU()
        
    else:
        model = MSFMA_short(args.input_length, args.num_channel, args.model_width, args.dilation, args.output_nums, args.forecasting_module).HGSPU()
    
    for i in range(5):

        from data_loader import load_dataset

        X_train, y_train, _, _ = load_dataset(args.read_path, args.read_sig_path, i+1, SCHDB=True)  # load MADB : SCHDB=False
        
        print('CV = ', i)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit([X_train[:,:1250],X_train[:,1250:2500], X_train[:,:2500:3750], X_train[:,3750:5000], X_train[:,:5000:6250], X_train[:,6250:]], y_train, epochs = 20, batch_size = 16, callbacks = [callback])
        
        model.save(args.model_save_dir + modelName + '_' + str(i) + '_' + 'pred_5hr' + '.h5')

