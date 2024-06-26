import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import numpy as np
import pandas as pd
from random import shuffle
from tf.keras.models import load_model
import sys
from tensorflow.python.keras.callbacks import Callback, LearningRateScheduler
from MSF_long_term import *
from MSF_short_term import *
import argparse
from evaluate import *

parser = argparse.ArgumentParser()
parser.add_argument('--read_path', type=str, default='./')              # Read labeling data 
parser.add_argument('--read_sig_path', type=str, default='./')          # Read signal data
parser.add_argument('--batch_size', type=int, default=16)               # Batch size
parser.add_argument('--epochs', type=int, default=20)                   # Ephochs
parser.add_argument('--model_save_dir', type=str, default='./')         # Path of saving model
parser.add_argument('--model_name', type=str, default='bsMSF')          # Model name saved

parser.add_argument('--lr', type=float, default=0.00005)                # Learning rate
parser.add_argument('--output_nums', type=int, default=5)               # Number of multi-step 


args, unknown = parser.parse_known_args()
if __name__ == '__main':
    print(args, file=sys.stdout, flush=True)

    for i in range(5):

        from data_loader import load_dataset

        _, _, X_test, y_test = load_dataset(args.read_path, args.read_sig_path, i+1, SCHDB=True)  # load MADB : SCHDB=False
        
        print('CV = ', i)

        
        model = load_model(args.model_save_dir + modelName + '_' + str(i) + '_' + 'pred_5hr' + '.h5')
        
        

        result = model.predict([X_test[:,:1250],X_test[:,1250:2500], X_test[:,:2500:3750], X_test[:,3750:5000], X_test[:,:5000:6250], X_test[:,6250:]])

        eval_pred(y_test, result,  i, args.output_nums, args.model_save_dir, args.model_name)