# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:56:40 2021

@author: RamiElRashid
"""

from fplearn.processing import compile_process_segmented_data
from tensorflow.keras.models import load_model

import numpy as np, pandas as pd

modelpath = 'C:/Data/ML/Y---/'
base_path = "C:/Users/RamiElRashid/Documents/fpCache/"

duoname = 'aphidvsall.h5'
trioname = 'aphid3way.h5'

model_duo = load_model(modelpath + duoname)
model_trio = load_model(modelpath + trioname)

#%%
from keras.utils import np_utils

sessid = 631
maxi = 1000
data_length = 1000

exdata, exlabels, _, _, exmids, _, _ = \
    compile_process_segmented_data([sessid], base_path, maxi, 
                                         data_length=data_length, num_channels=1)
Xex = exdata.reshape([-1, data_length, 1]).astype(np.float32)
Yex = np_utils.to_categorical(exlabels, 3).astype(np.float32)

pred2 = model_duo.predict(Xex)
pred3 = model_trio.predict(Xex)

predf = pd.DataFrame(np.concatenate([pred2, pred3], axis=1), columns=
                   ["Non-aphid", "Aphid", "Other insect", "BB", "PP"],
                   index=exmids)
#%%
print(predf.mean())
ands = predf[(predf['Non-aphid'] > .5) & (predf['Other insect'] > .5)]
ors = predf[(predf['Non-aphid'] > .5) | (predf['Other insect'] > .5)]
print("Definitely not aphids:", len(ands) / len(Xex))
print("Probably not aphids:  ", len(ors) / len(Xex))

#%%
import fpmodules as fp
# features = fp.get_features(measurementid=exmids, featureid=52)
# feat = features[features['WBF_SGO_combined'] > 0]

# ftgr = feat.groupby(['MeasurementId']).median()['WBF_SGO_combined']
# ftm = 
# len(ftgr[(ftgr > 80) & (ftgr < 130)]) / len(ftgr)
#%%
#from wbf import find_fundamental_sgo_combined
evs = fp.EventList(exmids).fill()
#%%
wbfs, maxes = [], []
for event in evs:
    wbs = np.array([find_fundamental_sgo_combined(ch) for ch in event.data])
    wbf = np.median(wbs[wbs > 0])
    wbfs.append(wbf)
    maxes.append(np.max(event.data))

wbfser = pd.Series(wbfs, index=evs.id, name="WBF")
maxser = pd.Series(maxes, index=evs.id, name="Max")
pwdf = predf.merge(wbfser, left_index=True, right_index=True).merge(maxser, left_index=True, right_index=True)

#ands = pwdf[(pwdf['Non-aphid'] > .5) & (pwdf['Other insect'] > .5)]
wors = pwdf[((pwdf['Non-aphid'] < .5) & (pwdf['Other insect'] < .5)) & ((pwdf['WBF'] > 80) & (pwdf['WBF'] < 130))]
print("In-wbf aphids - REMAINDER:", len(wors) / len(Xex))

lens = wors[wors['Max'] > 50]
print("WBF & max-restricted aphids - REMAINDER:", len(lens) / len(Xex))

print("TOTAL INSECTS: ", len(set(exmids)))
print("Aphid final ratio: ", len(set(wors.index)) / len(set(exmids)))
print("Aphid maxres ratio:", len(set(lens.index)) / len(set(exmids)))
#%% 
aphmids = list(set(wors.index))
avi = fp.EventList(aphmids)
