
def eval_pred(y_test_onehot, predict, i, out_num, outpath, name):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.metrics import confusion_matrix, classification_report
    y_grt = np.zeros((predict.shape[0], out_num))
    y_predanswer = np.zeros((predict.shape[0], out_num))
    for rhy in range(predict.shape[0]):
        recogt = np.reshape(predict[rhy, :], [1,predict.shape[1],2])
        y_grt[rhy, :] = np.argmax(y_test_onehot[rhy], axis = 1)
        # modelpred = loadmodel.predict(recogt)
        y_predanswer[rhy, :] = np.argmax(recogt[0,:,:], axis=1)

    for s in range(out_num):
        _step = confusion_matrix(y_grt[:,s], y_predanswer[:,s])
        if not os.path.exists(outpath+ '/confusion/'):
            os.makedirs(outpath+ '/confusion/')  
        pd.DataFrame(_step).to_csv(outpath + '/confusion/' + name + str(i) + '_step_' + str(s) + '.csv', index = False)
        if not os.path.exists(outpath+ '/classification_report/'):
            os.makedirs(outpath+ '/classification_report/') 
        pd.DataFrame(classification_report(y_grt[:,s], y_predanswer[:,s], output_dict=True)).transpose().to_csv(outpath + '/classification_report/' + name + str(i) + '_step_' + str(s) + '.csv')
