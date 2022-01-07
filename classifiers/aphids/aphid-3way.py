# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:47:58 2021

@author: Server1
"""
import fpmodules as fp
import numpy as np
import os
    
from fplearn.processing import compile_process_segmented_data

base_path = "C:/Users/RamiElRashid/Documents/fpCache/"
classes_short = ['Other insect', 'Black bean aphid', 'Peach potato aphid']


from fplearn.processing import mlready_data
from fplearn.run import run_model, evaluate_model

grouplist = [[363, 125, 777, 409,  # other coleoptera
    174, 520, 548, 706, 475, 471,          # ladybirds
    537, 206, 456, 331, 464, 162, 716, 686, 461],   # diversity quota
             
    [330, 156, 473, 173, 220, 513], # bb-aphid
    # 220 FOR VALIDATION
    
    [166, 466, 489, 225, 467, 476]] # pp-aphid
    # 225 FOR VALIDATION

# maxlist = [2000, 2000, 2000, 2000,
#            2000, 2000, 2000, 2000, 2000, 2000,
#            1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500,
#            2000,#
maxlist = [2000, 10000, 10000]
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
                          batch_size=200, epochs=35, stop_patience=7)

evals = evaluate_model(model, model.history.history, 
                       Xe.astype(np.float32), Ye.astype(np.float32),
                       classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%% I/O 
#%%------ DANGEROUS !!!!!!!!!! CHECK BEFORE F5-RUNNING 
# import json
# modelname = "aphid-3way"
# model.save(base_path + modelname + ".h5", save_format='h5')
# params_json = json.dumps(str(params))
# with open(base_path + modelname + "_params.json", "w") as fh:
#     fh.write(params_json)


# from tensorflow.keras.models import load_model
# model = load_model(base_path + modelname + ".h5")

#%% Test on field sessions

from keras.utils import np_utils  # we need to get rid of this

sessid = 631
maxi = 1000
ylen = 3

exdata, exlabels, _, _, exmids, _, _ = \
    compile_process_segmented_data([sessid], base_path, maxi, 
                                         data_length=data_length, num_channels=1)
Xex = exdata.reshape([-1, data_length, 1]).astype(np.float32)
Yex = np_utils.to_categorical(exlabels, ylen)

exeval = evaluate_model(model, None, Xex, Yex, classes_short, lossplot=False)

#%% test sensitivities
import matplotlib.pyplot as plt

maxi = 2000
ylen = 2

lengths = []
#sessids = grouplist[0] + grouplist[1] + [173, 220] + grouplist[2] + [388, 225]
sessids = grouplist[0] + grouplist[1] + [513] + grouplist[2] + [489]
misclass_bb, misclass_pp = [], []
for sessid in sessids:
    exdata, exlabels, _, _, exmids, _, _ = \
        compile_process_segmented_data([sessid], base_path, maxi, 
                                             data_length=data_length, num_channels=1)
    Xex = exdata.reshape([-1, data_length, 1])
    Yex = np_utils.to_categorical(exlabels, ylen)
    predix_vals = model.predict(Xex)
    predix = np.argmax(predix_vals, axis=1)
    misbb = len(np.where(predix ==1)[0]) / len(Xex) # misclass as bb
    mispp = len(np.where(predix ==2)[0]) / len(Xex) # misclass as pp
    
    lengths.append(Yex.shape[0])
    misclass_bb.append(misbb)
    misclass_pp.append(mispp)
    


import pandas as pd
results = pd.DataFrame({"SessionId": sessids, 
                        "Data points": lengths,
                        "Other insect": [1-(b+p) for (b,p) in zip(misclass_bb, misclass_pp)],
                        "Black Bean": misclass_bb,
                        "Peach Potato": misclass_pp})

results.to_csv(base_path + "aphid3_sens-inv-all-validation-513-489.csv")

