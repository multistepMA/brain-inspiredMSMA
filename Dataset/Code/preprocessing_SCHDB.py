import argparse
import wfdb
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

#################################################################

## Function List

# _read_ann: read label data
# _reproduce_label: converting label data
# apply_filters: filtering signal
# get_labels_start_end_time: extraction of start and end point in signal and label
# _rev_arrhythmia_type: identification arrhythmia type in patient
# _pred_zone_info: label malignant arrhythmia 

#################################################################

def _read_ann(ann_addpath, record_sig_len, vfonset):
    data = wfdb.rdann(ann_addpath, 'ari')
    _r_change = np.where(np.array(data.symbol) == '+')[0]
    _r_change_pos_tran = data.sample[_r_change]
    _r_type = np.array(data.aux_note)[_r_change]
    _pos_vf = np.where(_r_change_pos_tran > vfonset)[0]
    if len(_pos_vf) > 0:
        _r_change_pos_tran = np.insert(_r_change_pos_tran, _pos_vf[0], vfonset)
        _r_type = np.insert(_r_type, _pos_vf[0], '(VF')
    else:
        _r_change_pos_tran = np.insert(_r_change_pos_tran, len(_r_change_pos_tran), vfonset)
        _r_type = np.insert(_r_type, len(_r_type), '(VF')

    _r = np.where(np.array(data.symbol) != '+')[0]
    _r_pos_tran = data.sample[_r]

    if len(_r_change) > 0:
        if len(_r_type) > 0:
            _diff = np.hstack([np.diff(np.array(_r_change_pos_tran)), np.array(record_sig_len - np.array(_r_change_pos_tran)[-1])])
            _pos = np.array(_r_change_pos_tran)
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
    from scipy.signal import butter, filtfilt, detrend
    _nanmean = np.nanmean(ecgs)
    _ecgs = np.nan_to_num(ecgs, nan=_nanmean)
    b, a = butter(3, [0.5/(fs/2), 30/(fs/2)], 'bandpass')
    ecg_detrend = filtfilt(b, a, _ecgs)  # filtering
    ecg_detrend = detrend(ecg_detrend)
    ecg_norm = (ecg_detrend - np.min(ecg_detrend)) / (np.max(ecg_detrend) - np.min(ecg_detrend))

    return ecg_norm

def get_labels_start_end_time(frame_wise_labels):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    labels.append(frame_wise_labels[0])
    starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            labels.append(frame_wise_labels[i])
            starts.append(i)
            ends.append(i)
            last_label = frame_wise_labels[i]
    ends.append(i + 1)
    return labels, starts, ends

def _calc_vfonset(_vfonsetinfo, _sub_no):
    _index_no = np.where(_vfonsetinfo['Singlas'].values == int(_sub_no))[0]
    if _vfonsetinfo['VF Onset'].loc[_index_no[0]] != 'remove' and _vfonsetinfo['VF Onset'].loc[_index_no[0]] != '(no VF)':
        _vfonset_h = int(_vfonsetinfo['VF Onset'].loc[_index_no[0]].split(':')[0])
        _vfonset_m = int(_vfonsetinfo['VF Onset'].loc[_index_no[0]].split(':')[1])
        _vfonset_s = int(_vfonsetinfo['VF Onset'].loc[_index_no[0]].split(':')[2])

        _calc_h = ((_vfonset_h * 60) * 60) * 250
        _calc_m = (_vfonset_m * 60) * 250
        _calc_s = _vfonset_s * 250
        _sum_cacl_onset = _calc_h + _calc_m + _calc_s
    elif _vfonsetinfo['VF Onset'].loc[_index_no[0]] == '(no VF)':
        _sum_cacl_onset = 0
    else:
        _sum_cacl_onset = 0

    return _sum_cacl_onset

def _rev_arrhythmia_type(arrhythmia_type):
    _rev_arr_type = []
    for _a in range(len(arrhythmia_type)):
        _rev_arr_type.append(''.join(c for c in arrhythmia_type[_a] if c not in '('))
    return np.array(_rev_arr_type)

def _pred_zone_info(_step_label, _pred_first_zone, _pred_second_zone, labeldict):
    labels, starts, ends = get_labels_start_end_time(np.array(_step_label[_pred_first_zone:_pred_second_zone]))

    if len(np.where(np.array(labels) == labeldict['VT'])[0]) > 0 or len(np.where(np.array(labels) == labeldict['VF'])[0]) > 0:
        _sudden_loc = np.array(np.where(np.array(labels) == labeldict['VT'])[0] or np.where(np.array(labels) == labeldict['VF'])[0])
        _dur = np.sum(np.array(ends)[_sudden_loc] - np.array(starts)[_sudden_loc])
        if _dur >= (250*30):
            _pred_label_sud = 1
            _pred_label_nonsud = 0
        else:
            _pred_label_sud = 0
            _pred_label_nonsud = 0
    elif len(np.where(np.array(labels) == labeldict['AFIB'])[0]) > 0:        
        _pot_sudden_loc = np.where(np.array(labels) == labeldict['AFIB'])[0]
        _dur = np.sum(np.array(ends)[_pot_sudden_loc] - np.array(starts)[_pot_sudden_loc])
        if _dur >= (250*30):
            _pred_label_nonsud = 1
        else:
            _pred_label_nonsud = 0
        _pred_label_sud = 0
        _pred_label_safe = 0
    else:
        _pred_label_sud = 0
        _pred_label_nonsud = 0
        _pred_label_safe = 1

    if (_pred_label_sud == 1 and _pred_label_nonsud == 1) or (_pred_label_sud == 1 and _pred_label_nonsud == 0):
        _pred_label = 2
    elif _pred_label_sud == 0 and _pred_label_nonsud == 1:
        _pred_label = 1
    else:
        _pred_label = 0

    return _pred_label

def main(path, outpath):
    labeldict = dict({'AFIB': 100, 'AFL': 101, 'J': 145, 'B': 133, 'T': 134, 'VT': 110, 'SVTA': 103, 'NOD': 999, 'IVR': 143, 
        'VFL': 112, 'VF': 112,'TS': 888, 'P': 777, 'MISS': 132, 'AB': 666, 'PREX': 555, 'BII': 444, 'SBR': 333, 'N': -100})

    sig_addpath = sorted(glob.glob(path + '/*.dat'))
    _vfonset_info = pd.read_csv(path + 'SCDHDB_time_info.csv')
    
    _m = [26, 13, 18, 7, 0, 0, 8, 10, 8, 0, 0, 0, 18, 1, 0, 0, 9, 44, 7, 6, 0, 0, 5]    
    _s = [35, 20, 0, 0, 0, 0, 40, 0, 55, 0, 0, 0, 10, 20, 0, 40, 50, 10, 45, 5, 22, 0, 55]

    for sub in range(len(sig_addpath)):
        print(os.path.basename(sig_addpath[sub])[:-4])
        record = wfdb.rdrecord(sig_addpath[sub][:-4]).p_signal[:, 0]

        _vfonset = _calc_vfonset(_vfonset_info, os.path.basename(sig_addpath[sub])[:-4])    
        _sub_diff, _sub_pos, _sub_arrhythmia_type = _read_ann(path + os.path.basename(sig_addpath[sub])[:-4], len(record), _vfonset)
        print(np.unique(_sub_arrhythmia_type))
                
        _sub_sig = apply_filters(record, 250)

        _rev_sub_arrhythmia_type = _rev_arrhythmia_type(_sub_arrhythmia_type)
        _step_label = _reproduce_label(_rev_sub_arrhythmia_type, _sub_pos, _sub_diff, len(record), labeldict)

        _mov_first_zone = 30*250
        _mov_last_zone = 120*60*250
        _begin = (_m[sub]*60*250) + (_s[sub]*250)

        if _vfonset > 0 and _vfonset_info['VF Onset'][sub] != 'remove':
            for _vf in tqdm(range(_mov_first_zone, _mov_last_zone, 250)):
                _pred_vf_zone = (_vfonset + _vf) - (10*60*250)
                _pred_first_zone = _pred_vf_zone - (300*60*250)
                _pred_second_zone = _pred_vf_zone - (240*60*250)
                _pred_third_zone = _pred_vf_zone - (180*60*250)
                _pred_fourth_zone = _pred_vf_zone - (120*60*250)

                if _pred_first_zone < 0:
                    time = 0
                elif _pred_first_zone < 0:
                    time = 0

                if (_pred_fourth_zone + (60*60*250)) < len(record) and (_pred_first_zone > 0):
                    _pred_output1 = _pred_zone_info(_step_label, _pred_first_zone, _pred_second_zone, labeldict)
                    _pred_output2 = _pred_zone_info(_step_label, _pred_second_zone, _pred_third_zone, labeldict)
                    _pred_output3 = _pred_zone_info(_step_label, _pred_third_zone, _pred_fourth_zone, labeldict)
                    _pred_output4 = _pred_zone_info(_step_label, _pred_fourth_zone, _pred_vf_zone, labeldict)
                    _pred_output5 = _pred_zone_info(_step_label, _pred_vf_zone, _pred_vf_zone + (60*60*250), labeldict)

                    if _pred_first_zone - (10*60*250) > 0:
                        _class_first_zone = _pred_first_zone - (10*60*250)
                        _arr_num_pos, _, _ = get_labels_start_end_time(np.array(_step_label[_class_first_zone:(_class_first_zone + 250*30)]))   
                        if len(_sub_sig[_class_first_zone:(_class_first_zone + 250*30)]) > ((250*30) - 10):     
                            if not os.path.exists(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'VF'):           
                                os.makedirs(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'VF')        
                                os.makedirs(outpath + 'Pred/'+ os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'VF') 
                            pd.DataFrame([_sub_sig[_class_first_zone:(_class_first_zone + 250*30)], _step_label[_class_first_zone:(_class_first_zone + 250*30)]]).transpose().to_csv(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4] + '/'  + 'VF' + '/' + os.path.basename(sig_addpath[sub])[:-4] + '_' + str(_class_first_zone) + '.csv')
                            pd.DataFrame([_pred_output1, _pred_output2, _pred_output3, _pred_output4, _pred_output5]).transpose().to_csv(outpath + 'Pred/'+ os.path.basename(sig_addpath[sub])[:-4] + '/'  + 'VF' + '/' + os.path.basename(sig_addpath[sub])[:-4] + '_' + str(_class_first_zone)+ str(np.array([_pred_output1, _pred_output2, _pred_output3, _pred_output4, _pred_output5])) + '.csv')
        
        elif _vfonset == 0 and _vfonset_info['VF Onset'][sub] == '(no VF)':
            for _no_vf in tqdm(range(_begin, len(record), 1250)):
                _class_first_zone = _no_vf

                _pred_no_first_zone = _class_first_zone + (30*250) + (10*60*250)
                _pred_no_second_zone = _pred_no_first_zone + (60*60*250)
                _pred_no_third_zone = _pred_no_second_zone + (60*60*250)
                _pred_no_fourth_zone = _pred_no_third_zone + (60*60*250)
                _pred_no_fifth_zone = _pred_no_fourth_zone + (60*60*250)

                if (_pred_no_first_zone + (30*60*250)) < len(record) and (_pred_no_fifth_zone + (60*60*250) < len(record)):
                    _pred_no_output1 = _pred_zone_info(_step_label, _pred_no_first_zone, _pred_no_second_zone, labeldict)
                    _pred_no_output2 = _pred_zone_info(_step_label, _pred_no_second_zone, _pred_no_third_zone, labeldict)
                    _pred_no_output3 = _pred_zone_info(_step_label, _pred_no_third_zone, _pred_no_fourth_zone, labeldict)
                    _pred_no_output4 = _pred_zone_info(_step_label, _pred_no_fourth_zone, _pred_no_fifth_zone, labeldict)
                    _pred_no_output5 = _pred_zone_info(_step_label, _pred_no_fifth_zone, _pred_no_fifth_zone + (60*60*250), labeldict)

                    _arr_num_pos, _, _ = get_labels_start_end_time(np.array(_step_label[_class_first_zone:(_class_first_zone + 250*30)]))   
                    if len(_sub_sig[_class_first_zone:(_class_first_zone + 250*30)]) > ((250*30) - 10):     
                        if not os.path.exists(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4]+ '/'  + 'NoVF'):           
                            os.makedirs(outpath+ 'Classification/' + os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'NoVF')        
                            os.makedirs(outpath+ 'Pred/' + os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'NoVF')        
                        pd.DataFrame([_sub_sig[_class_first_zone:(_class_first_zone + 250*30)], _step_label[_class_first_zone:(_class_first_zone + 250*30)]]).transpose().to_csv(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4] + '/' + 'NoVF' + '/' + os.path.basename(sig_addpath[sub])[:-4] + '_' + str(_class_first_zone) + '.csv')
                        pd.DataFrame([_pred_no_output1, _pred_no_output2, _pred_no_output3, _pred_no_output4, _pred_no_output5]).transpose().to_csv(outpath + 'Pred/'+ os.path.basename(sig_addpath[sub])[:-4] + '/' + 'NoVF' + '/' + os.path.basename(sig_addpath[sub])[:-4] + '_' + str(_class_first_zone)+ str(np.array([_pred_no_output1, _pred_no_output2, _pred_no_output3, _pred_no_output4, _pred_no_output5])) + '.csv')

        print('==============VF zone Done')

        _pred_first_zone_no = _vfonset - (90*60*250)

        for _no_vf in tqdm(range(_begin, _pred_first_zone_no, 1250)):
            _class_first_zone = _no_vf

            _pred_no_first_zone = _class_first_zone + (30*250) + (10*60*250)
            _pred_no_second_zone = _pred_no_first_zone + (60*60*250)
            _pred_no_third_zone = _pred_no_second_zone + (60*60*250)
            _pred_no_fourth_zone = _pred_no_third_zone + (60*60*250)
            _pred_no_fifth_zone = _pred_no_fourth_zone + (60*60*250)
            _pred_no_sixth_zone = _pred_no_fifth_zone + (60*60*250)
            _pred_no_seventh_zone = _pred_no_sixth_zone + (60*60*250)

            if (_pred_no_first_zone + (30*60*250)) < len(record) and (_pred_no_seventh_zone + (60*60*250) < len(record)):
                _pred_no_output1 = _pred_zone_info(_step_label, _pred_no_first_zone, _pred_no_second_zone, labeldict)
                _pred_no_output2 = _pred_zone_info(_step_label, _pred_no_second_zone, _pred_no_third_zone, labeldict)
                _pred_no_output3 = _pred_zone_info(_step_label, _pred_no_third_zone, _pred_no_fourth_zone, labeldict)
                _pred_no_output4 = _pred_zone_info(_step_label, _pred_no_fourth_zone, _pred_no_fifth_zone, labeldict)
                _pred_no_output5 = _pred_zone_info(_step_label, _pred_no_fifth_zone, _pred_no_fifth_zone + (60*60*250), labeldict)

                _arr_num_pos, _, _ = get_labels_start_end_time(np.array(_step_label[_class_first_zone:(_class_first_zone + 250*30)]))   
                if len(_sub_sig[_class_first_zone:(_class_first_zone + 250*30)]) > ((250*30) - 10):     
                    if not os.path.exists(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4]+ '/'  + 'NoVF'):           
                        os.makedirs(outpath+ 'Classification/' + os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'NoVF')        
                        os.makedirs(outpath+ 'Pred/' + os.path.basename(sig_addpath[sub])[:-4]+ '/' + 'NoVF')      
                    pd.DataFrame([_sub_sig[_class_first_zone:(_class_first_zone + 250*30)], _step_label[_class_first_zone:(_class_first_zone + 250*30)]]).transpose().to_csv(outpath + 'Classification/'+ os.path.basename(sig_addpath[sub])[:-4] + '/' + 'NoVF' + '/' + os.path.basename(sig_addpath[sub])[:-4] + '_' + str(_class_first_zone) + '.csv')
                    pd.DataFrame([_pred_no_output1, _pred_no_output2, _pred_no_output3, _pred_no_output4, _pred_no_output5]).transpose().to_csv(outpath + 'Pred/'+ os.path.basename(sig_addpath[sub])[:-4] + '/' + 'NoVF' + '/' + os.path.basename(sig_addpath[sub])[:-4] + '_' + str(_class_first_zone)+ str(np.array([_pred_no_output1, _pred_no_output2, _pred_no_output3, _pred_no_output4, _pred_no_output5])) + '.csv')

        print('==============No VF zone Done')
        print('Make csv!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SCHDB data")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing SCHDB data files") 
    parser.add_argument("--outpath", type=str, required=True, help="Path to save the preprocessed SCHDB data") 

    args = parser.parse_args()
    main(args.path, args.outpath)