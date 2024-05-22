import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

def onehot(y_data, _shape):
    enc = OneHotEncoder()
    y_data_onehot = []

    for xx in range(y_data.shape[0]):
        enc.fit(np.array([0, 1]).reshape(-1, 1))
        annotat_onehot = enc.transform(np.reshape(y_data[xx], (_shape, 1))).toarray()
        annotat_onehot = smooth_labels(annotat_onehot, factor=0.2)
        y_data_onehot.append(annotat_onehot)

    return np.array(y_data_onehot)

def load_dataset(_read_path, _read_sig_path, cv_num, SCHDB=True):
    
    if SCHDB:
        no_vf_files = []
        vf_files = []

        for sub in tqdm(sorted(os.listdir(_read_sig_path)), desc="Reading patient folders"):
            for condition in ['NoVF', 'VF']:
                path = os.path.join(_read_sig_path, sub, condition)
                if condition == 'NoVF':
                    no_vf_files.extend([(sub, condition, f) for f in sorted(os.listdir(path))])
                else:
                    vf_files.extend([(sub, condition, f) for f in sorted(os.listdir(path))])

        no_vf_files = np.array(no_vf_files)
        vf_files = np.array(vf_files)

        kf = KFold(n_splits=5, random_state=1004, shuffle=True)

        no_vf_train_indices, no_vf_test_indices = list(kf.split(no_vf_files))[cv_num - 1]
        vf_train_indices, vf_test_indices = list(kf.split(vf_files))[cv_num - 1]

        train_files = np.concatenate((no_vf_files[no_vf_train_indices], vf_files[vf_train_indices]))
        test_files = np.concatenate((no_vf_files[no_vf_test_indices], vf_files[vf_test_indices]))
    
    else:
        files = []

        for sub in tqdm(sorted(os.listdir(_read_sig_path)), desc="Reading patient folders"):
            path = os.path.join(_read_sig_path, sub)
            files.extend([(sub, f) for f in sorted(os.listdir(path))])

        files = np.array(files)

        kf = KFold(n_splits=5, random_state=1004, shuffle=True)
        
        train_indices, test_indices = list(kf.split(files))[cv_num - 1]

        train_files = files[train_indices]
        test_files = files[test_indices]
    
    X_train, y_train = _load_data(_read_path, _read_sig_path, train_files, "Loading train data", SCHDB)
    X_test, y_test = _load_data(_read_path, _read_sig_path, test_files, "Loading test data", SCHDB)
    

    y_train[y_train == 2] = 1
    y_test[y_test == 2] = 1

    y_train = onehot(y_train, y_train.shape[1])
    y_test = onehot(y_test, y_test.shape[1])
    
    return X_train, y_train, X_test, y_test

def _load_data(_read_path, _read_sig_path, files, desc, SCHDB):
    X_data = None
    y_data = None

    for item in tqdm(files, desc=desc):
        if SCHDB:
            sub, condition, file_name = item
            temp = pd.read_csv(os.path.join(_read_sig_path, sub, condition, file_name))
            x_data_part = temp['0'].to_numpy()
            
            base_name = file_name.split('.csv')[0]
            matching_file = [f for f in os.listdir(os.path.join(_read_path, sub, condition)) if f.split('[')[0] == base_name][0]
            y_values = matching_file.split('[')[1].split(']')[0].split()
        else:
            sub, file_name = item
            temp = pd.read_csv(os.path.join(_read_sig_path, sub, file_name))
            x_data_part = temp['0'].to_numpy()

            base_name = file_name.split('.csv')[0]
            matching_file = [f for f in os.listdir(os.path.join(_read_path, sub)) if f.split('[')[0] == base_name][0]
            y_values = matching_file.split('[')[1].split(']')[0].split()
        y_data_part = np.array(y_values, dtype=int)

        if X_data is None:
            X_data = x_data_part.reshape(1, -1)
            y_data = y_data_part.reshape(1, -1)
        else:
            X_data = np.vstack((X_data, x_data_part))
            y_data = np.vstack((y_data, y_data_part))

    return X_data, y_data

