# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:47:58 2021

@author: Server1
"""
import fpmodules as fp
import numpy as np
import os
    
from fplearn.processing import compile_process_segmented_data

base_path = "Y:/fpCache/ML/Events/"
classes_short = ['Not ppaphid', 'Peach potato aphid']


from fplearn.processing import mlready_data
from fplearn.run import run_model, evaluate_model

grouplist = [[363, 125, 777, 409,  # other coleoptera
    174, 520, 548, 706, 475, 471,          # ladybirds
    537, 206, 456, 331, 464, 162, 716, 686, 461,   # diversity quota
    220, 330, 173, 156, 473, 513], # bb-aphid
             [166, 225, 466, 467, 489, 388, 476]] # pp-aphid

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
                          batch_size=200, epochs=70, stop_patience=10)

evals = evaluate_model(model, model.history.history, Xe, Ye, classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%% I/O 
#%%------ DANGEROUS !!!!!!!!!! CHECK BEFORE F5-RUNNING 
# import json
# modelname = "ppaphid-bs1-fl"
# model.save(base_path + modelname + ".h5", save_format='h5')
# params_json = json.dumps(str(params))
# with open(base_path + modelname + "_params.json", "w") as fh:
#     fh.write(params_json)


# from tensorflow.keras.models import load_model
# model = load_model(base_path + modelname + ".h5")

#%% Test on field sessions

from keras.utils import np_utils  # we need to get rid of this

sessid = 628
maxi = 1000
ylen = 2

exdata, exlabels, _, _, exmids, _, _ = \
    compile_process_segmented_data([sessid], base_path, maxi, 
                                         data_length=data_length, num_channels=1)
Xex = exdata.reshape([-1, data_length, 1])
Yex = np_utils.to_categorical(exlabels, ylen)

exeval = evaluate_model(model, None, Xex, Yex, classes_short, lossplot=False)

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

results.to_csv(base_path + "ppaphid1_sensitivities1.csv")

