import wfdb
import glob
import numpy as np
import os
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.signal import butter, filtfilt, detrend


#################################################################

## Function List

# _read_ann: read label data
# _reproduce_label: converting label data
# apply_filters: filtering signal
# get_labels_start_end_time: extraction of start and end point in signal and label
# _rev_arrhythmia_type: identification arrhythmia type in patient
# _pred_zone_info: label malignant arrhythmia 

#################################################################

def _read_ann(ann_addpath, record_sig_len):
    data = wfdb.rdann(ann_addpath, 'atr')

    _r_change = np.where(np.array(data.symbol) == '+')[0]
    _r_change_pos_tran = data.sample[_r_change]
    _r_type = np.array(data.aux_note)[_r_change]

    _r = np.where(np.array(data.symbol) != '+')[0]
    _r_pos_tran = data.sample[_r]

    if len(_r_change) > 0:
        if len(_r_type) > 0:
            _diff = np.hstack([np.diff(np.array(_r_change_pos_tran)), np.array(record_sig_len - np.array(_r_change_pos_tran)[-1])])
            _pos = np.array(_r_change_pos_tran)
        else:
            _diff = []
            _pos = []
    else:
        _diff = []
        _pos = []

    return _diff, _pos, _r_type

def _reproduce_label(_sub_arrhythmia_type, _sub_pos, _sub_diff, record_sig_len, labeldict):
    if _sub_arrhythmia_type[0] != 'N':
        _first_label = np.tile(labeldict['N'], _sub_pos[0])
    else:
        _first_label = np.tile(labeldict[_sub_arrhythmia_type[0]], _sub_pos[0])
    
    _temp_label_tot = np.zeros((1,))
    for _temp in range(len(_sub_arrhythmia_type)):
        _temp_label_tot = np.concatenate([_temp_label_tot, np.tile(labeldict[_sub_arrhythmia_type[_temp]], _sub_diff[_temp])], axis=0)
    
    _last_label = np.tile(labeldict[_sub_arrhythmia_type[-1]], record_sig_len - (_sub_diff.sum() + _sub_pos[0]))

    return np.hstack([_first_label, _temp_label_tot[1:], _last_label])

def apply_filters(ecgs, fs):
    _nanmean = np.nanmean(ecgs)
    _ecgs = np.nan_to_num(ecgs, nan=_nanmean)
    b, a = butter(3, [0.5 / (fs / 2), 30 / (fs / 2)], 'bandpass')
    ecg_detrend = filtfilt(b, a, _ecgs)  
    ecg_detrend = detrend(ecg_detrend)
    ecg_norm = (ecg_detrend - np.min(ecg_detrend)) / (np.max(ecg_detrend) - np.min(ecg_detrend))

    return ecg_norm

def get_labels_start_end_time(frame_wise_labels):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    labels.append(last_label)
    starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            labels.append(frame_wise_labels[i])
            starts.append(i)
            ends.append(i)
            last_label = frame_wise_labels[i]
    ends.append(i + 1)
    return labels, starts, ends


def _rev_arrhythmia_type(arrhythmia_type):
    _rev_arr_type = []
    for _a in range(len(arrhythmia_type)):
        _rev_arr_type.append(''.join(c for c in arrhythmia_type[_a] if c not in '('))
    return np.array(_rev_arr_type)

def _pred_zone_info(_step_label, _pred_first_zone, _pred_second_zone, labeldict):
    labels, starts, ends = get_labels_start_end_time(np.array(_step_label[_pred_first_zone:_pred_second_zone]))
    pred_label_sud, pred_label_nonsud, pred_label_safe = 0, 0, 1

    if len(np.where(np.array(labels) == labeldict['VT'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['VFL'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['VF'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['VFIB'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['ASYS'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['VER'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['HGEA'])[0]) > 0:
        _sudden_loc = np.concatenate([np.where(np.array(labels) == labeldict['VT'])[0], np.where(np.array(labels) == labeldict['VFL'])[0], np.where(np.array(labels) == labeldict['VFIB'])[0],np.where(np.array(labels) == labeldict['VF'])[0], np.where(np.array(labels) == labeldict['ASYS'])[0], np.where(np.array(labels) == labeldict['VER'])[0], np.where(np.array(labels) == labeldict['HGEA'])[0]])
        _sudden_loc = np.unique(_sudden_loc)
        _dur = np.sum(np.array(ends)[_sudden_loc] - np.array(starts)[_sudden_loc])
        if _dur > (250*30):
            pred_label_sud = 1
            pred_label_nonsud = 0
        else:
            pred_label_sud = 0
            pred_label_nonsud = 0
    elif len(np.where(np.array(labels) == labeldict['AFIB'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['B'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['BI'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['NOD'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['SBR'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['SVTA'])[0]) > 0:        
        _pot_sudden_loc = np.concatenate([np.where(np.array(labels) == labeldict['AFIB'])[0], np.where(np.array(labels) == labeldict['B'])[0], np.where(np.array(labels) == labeldict['BI'])[0], np.where(np.array(labels) == labeldict['NOD'])[0], np.where(np.array(labels) == labeldict['SBR'])[0], np.where(np.array(labels) == labeldict['SVTA'])[0]])
        _pot_sudden_loc = np.unique(_pot_sudden_loc)
        _pot_dur = np.sum(np.array(ends)[_pot_sudden_loc] - np.array(starts)[_pot_sudden_loc])
        if _pot_dur > (250*30):
            pred_label_nonsud = 1
        else:
            pred_label_nonsud = 0
        pred_label_sud = 0
        pred_label_safe = 0
    else:
        pred_label_sud = 0
        pred_label_nonsud = 0
        pred_label_safe = 1

    if (pred_label_sud == 1 and pred_label_nonsud == 1) or (pred_label_sud == 1 and pred_label_nonsud == 0):
        pred_label = 2
    elif pred_label_sud == 0 and pred_label_nonsud == 1:
        pred_label = 1
    else:
        pred_label = 0

    return pred_label

def main(path, outpath):
    labeldict = dict({
        'AFIB': 100, 'AFL': 101, 'J': 145, 'B': 133, 'T': 134, 'VT': 110, 'SVTA': 103, 'NOD': 888, 'IVR': 143, 'NOISE': 999,
        'VFL': 112, 'VF': 112, 'VFIB': 112, 'ASYS': 222, 'HGEA': 777, 'VER': 132, 'AB': 666, 'PREX': 555, 'BI': 444, 'SBR': 333, 'N': -100, 'NSR': -100
    })

    sig_addpath = sorted(glob.glob(os.path.join(path, '*.dat')))
    ann_addpath = sorted(glob.glob(os.path.join(path, '*.atr')))

    for sub in range(len(sig_addpath)):
        print(os.path.basename(sig_addpath[sub])[:-4])
        record = wfdb.rdrecord(sig_addpath[sub][:-4]).p_signal[:, 0]

        _sub_diff, _sub_pos, _sub_arrhythmia_type = _read_ann(ann_addpath[sub][:-4], len(record))
        print(np.unique(_sub_arrhythmia_type))

        _sub_sig = apply_filters(record, 250)

        _rev_sub_arrhythmia_type = _rev_arrhythmia_type(_sub_arrhythmia_type)
        _step_label = _reproduce_label(_rev_sub_arrhythmia_type, _sub_pos, _sub_diff, len(record), labeldict)

        _mov_first_zone = 0
        _mov_last_zone = len(record) - (30 * 60 * 250)

        for _vf in tqdm(range(_mov_first_zone, _mov_last_zone, 250)):
            _pred_first_zone = _vf + (5 * 60 * 250)
            _pred_second_zone = _vf + (10 * 60 * 250)
            _pred_third_zone = _vf + (15 * 60 * 250)
            _pred_forth_zone = _vf + (20 * 60 * 250)
            _pred_fifth_zone = _vf + (25 * 60 * 250)

            _pred_output1 = _pred_zone_info(_step_label, _pred_first_zone, _pred_second_zone, labeldict)
            _pred_output2 = _pred_zone_info(_step_label, _pred_second_zone, _pred_third_zone, labeldict)
            _pred_output3 = _pred_zone_info(_step_label, _pred_third_zone, _pred_forth_zone, labeldict)
            _pred_output4 = _pred_zone_info(_step_label, _pred_forth_zone, _pred_fifth_zone, labeldict)
            _pred_output5 = _pred_zone_info(_step_label, _pred_fifth_zone, len(record), labeldict)

            _class_first_zone = _vf
            _arr_num_pos, _, _ = get_labels_start_end_time(np.array(_step_label[_class_first_zone:(_class_first_zone + 250 * 30)]))
            if len(_sub_sig[_class_first_zone:(_class_first_zone + 250 * 30)]) > ((250 * 30) - 10):
                class_dir = os.path.join(outpath, 'Classification', os.path.basename(sig_addpath[sub])[:-4])
                pred_dir = os.path.join(outpath, 'Pred', os.path.basename(sig_addpath[sub])[:-4])
                os.makedirs(class_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)

                class_csv_path = os.path.join(class_dir, '{}_{}.csv'.format(os.path.basename(sig_addpath[sub])[:-4], _class_first_zone))
                pred_csv_path = os.path.join(pred_dir, '{}_{}{}.csv'.format(os.path.basename(sig_addpath[sub])[:-4], _class_first_zone, str(np.array([_pred_output1, _pred_output2, _pred_output3, _pred_output4, _pred_output5]))))

                pd.DataFrame([_sub_sig[_class_first_zone:(_class_first_zone + 250 * 30)], _step_label[_class_first_zone:(_class_first_zone + 250 * 30)]]).transpose().to_csv(class_csv_path)
                pd.DataFrame([_pred_output1, _pred_output2, _pred_output3, _pred_output4, _pred_output5]).transpose().to_csv(pred_csv_path)

        print('==============No VF zone Done')
        print('Make csv!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MADB data")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing MADB data files") 
    parser.add_argument("--outpath", type=str, required=True, help="Path to save the preprocessed MADB data") 

    args = parser.parse_args()
    main(args.path, args.outpath)
