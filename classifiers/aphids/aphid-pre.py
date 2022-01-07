# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:47:58 2021

@author: Server1
"""
import fpmodules as fp
import numpy as np
import os
    
from fplearn.processing import compile_process_segmented_data

#base_path = "Y:/fpCache/ML/Events/"
base_path = "C:/Users/RamiElRashid/Documents/fpCache/"
classes_short = ['Not aphid', 'Aphid']


from fplearn.processing import mlready_data
from fplearn.run import run_model, evaluate_model

grouplist = [[363, 125, 777, 409,  # other coleoptera
    174, 520, 548, 706, 475, 471,          # ladybirds
    537, 206, 456, 331, 464, 162, 716, 686, 461, 591, 385, 734, #diversity quota
    797, 787, 753, 773, 440, 32 ],   # 
             #MASSIVE
    [220, 330, 173, 156, 473, 513, # bb-aphid # removed 330
    166, 225, 466, 489, 476]] # pp-aphid # removed 388, 467

# maxlist = [2000, 2000, 2000, 2000,
#            2000, 2000, 2000, 2000, 2000, 2000,
#            1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500,
#            2000,#
maxlist = [2000, 15000]
data_length = 1000

#TRY THIS IN THREE-FOLD CLASSIFIER AS WELL
#%%

data, labels, files, raw, mid, seg, wave = \
    compile_process_segmented_data(grouplist, base_path, maxlist,
                                        split_channels=True, data_length=data_length,
                                        num_channels=1)

(Xt, Xv, Xe, Yt, Yv, Ye, \
 ft, fv, fe, rt, rv, re, mt, mv, me, st, sv, se, wt, wv, we), cw = \
    mlready_data(data, labels, files, raw, mid, seg, wave)
    
Xt = Xt.reshape([-1, data_length, 1])
Xv = Xv.reshape([-1, data_length, 1])
Xe = Xe.reshape([-1, data_length, 1])
    
model, params = run_model(Xt, Xv, Yt, Yv, class_weights=cw, 
                          batch_size=200, epochs=40, stop_patience=10)

evals = evaluate_model(model, model.history.history, 
                       Xe.astype(np.float32), Ye.astype(np.float32),
                       classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%% I/O 
#%%------ DANGEROUS !!!!!!!!!! CHECK BEFORE F5-RUNNING 
# import json
#modelname = "allaphid1"
#model.save(base_path + modelname + ".h5", save_format='h5')
# params_json = json.dumps(str(params))
# with open(base_path + modelname + "_params.json", "w") as fh:
#     fh.write(params_json)


# from tensorflow.keras.models import load_model
# model = load_model(base_path + modelname + ".h5")

#%% Test on field sessions

from keras.utils import np_utils  # we need to get rid of this
# a classifier like this (both aphids vs all) is shit on even training data
# too many false positives, but at least it picks out the aphids!
# we left out 467 and 330 and it gave 99%ish accuracy recognizing them
# 23 pb are 0.57...
# train on small-black-buggy things

# do wbf analysis
# session 388 is suspicios still here

sessid = 817
maxi = 1000
ylen = 2

exdata, exlabels, _, _, exmids, _, _ = \
    compile_process_segmented_data([sessid], base_path, maxi, 
                                         data_length=data_length, num_channels=1)
Xex = exdata.reshape([-1, data_length, 1]).astype(np.float32)
Yex = np_utils.to_categorical(exlabels, ylen)

exeval = evaluate_model(model, None, Xex, Yex, classes_short, lossplot=False)

#%% Check wbfs
predix = model.predict(Xex)
posit = np.where(predix[:,1] >= .5)[0]
negit = np.where(predix[:,0] >= .5)[0]
posmids = np.array(exmids)[posit]
negmids = np.array(exmids)[negit]

ftPos = fp.get_features(measurementid=list(posmids), featureid=52)
ftNeg = fp.get_features(measurementid=list(negmids), featureid=52)

fp.compare_hists([ftPos['WBF_SGO_combined'].tolist(), ftNeg['WBF_SGO_combined'].tolist()])


#%% test sensitivities
import matplotlib.pyplot as plt

maxi = 1000
ylen = 2

lengths = []
sessids = grouplist[0]
axx = []
for sessid in sessids:
    exdata, exlabels, _, _, exmids, _, _ = \
        compile_process_segmented_data([sessid], base_path, maxi, 
                                             data_length=data_length, num_channels=1)
    Xex = exdata.reshape([-1, data_length, 1])
    Yex = np_utils.to_categorical(exlabels, ylen)
    exeval = evaluate_model(model, None, Xex, Yex, classes_short, lossplot=False)
    lengths.append(Yex.shape[0])
    axx.append(exeval[-1][-1])
    
    # THIS IS BECAUSE THE CRAPPY THING HASN'T BEEN IMPLEMETNED YET
    plt.close('all')


import pandas as pd
results = pd.DataFrame({"SessionId": sessids, 
                        "Data points": lengths,
                        "Accuracy": axx})

results.to_csv(base_path + "aphidall_sensitivities1.csv")

