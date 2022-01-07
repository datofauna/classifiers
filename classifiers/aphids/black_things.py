# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:38:46 2021

@author: Server1
"""
# Big picture small black thingy comparison

base_path = "Y:/fpCache/ML/Events/"
classes_short = ['Pollen beetle', 'CSFB', 'Black bean aphid', 'Black adalia ladybird']
grouplist = [[777], [347, 363], [156, 173, 220, 330, 473, 513], [520]]
maxlist = [3000, 1500, 800, 10000]
data_length=1000

from fplearn.processing import compile_process_segmented_data, mlready_data
from fplearn.run import run_model, evaluate_model
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
    #%%
model, params = run_model(Xt, Xv, Yt, Yv, class_weights=cw, 
                          batch_size=200, epochs=70, stop_patience=12)

evals = evaluate_model(model, model.history.history, Xe, Ye, classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%%
from fplearn.tsne import TSNEplot
oom = 1000
tsne = TSNEplot(Xt[:oom], Yt[:oom], model, classes=classes_short)
tsne.fit()
tsne.plot()
