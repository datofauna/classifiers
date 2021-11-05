# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:38:46 2021

@author: Server1
"""
# Big picture small black thingy comparison
import numpy as np

base_path = "C:/Users/RamiElRashid/EventCache/Events/"
# classes_short = ['Cydia pomonella',# MALE',
#                  #'Cydia pomonella FEMALE',
#                  'Acleris comariana MALE',
#                  'Acleris comariana FEMALE',
#                  'Lobesia botrana']

# grouplist = [[591, 588],
#              [992, 943], [940, 989],
#              [162]]

# classes_short = ['D. suzukii', 'O. rufa', 'A. mellifera', 'hoverfly mix']
# grouplist = [498, 749, [686, 836], [1046, 1072]]

# classes_short = ['A. m. iberica', 'A. m. carnica', 'A. m. ligustica']
# grouplist = [[799, 817], [1052], [686, 805]]

#classes_short = ['Apis mellifera subsps.', 'Lookalikes']
# # B. pascuorum, D. suzukii, O. rufa
# grouplist = [[817,836,686,799,805,1052], [1082,498,749]]
classes_short = ['Not honeybee',# 'Bombus pascuorum', 
                 'Honeybee']
grouplist = [[749, 1072, 1046, 168, 171, 484, 478, 1139, ], 
    [817,836,686,799,805,1052, 1141]
             ]#, [527, 526]]#,498]
    # bombus pascuorum, all hoverflies, osmia rufa
data_length = 1000
maxlist = [30000 // len(x) for x in grouplist]


# classes_short = ['910', '989']
# grouplist = [940, 989]
# maxlist=1500

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
                          batch_size=500, epochs=30, stop_patience=12, learning_rate=0.0005)

evals = evaluate_model(model, model.history.history, Xe.astype(float), Ye, classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%%
from fplearn.tsne import TSNEplot
oom = 3000
tsne = TSNEplot(Xt[:oom], Yt[:oom], model, classes=classes_short)
tsne.fit()
tsne.plot()

#%%
from fpmodules.plotting.fancy import ScatterPicker

sc = ScatterPicker(tsne.tsne[:,0], tsne.tsne[:,1], Xt[:oom])
sc.display()

#%%
Ys = np.argmax(Yt[:oom], axis=1)
marker_dict = {0:'x', 1:'^', 2:'o', 3:'+'}
Ym = np.array([marker_dict[z] for z in Ys])
#cols = np.array([np.log(-np.min(x)) for x in Xt[:oom]])
#cols = np.array([fp.wbf(z.reshape(-1)) / 5 for z in Xt[:oom]]) # Xt wbf  (/5 for ds)
#cols = np.array([fp.wbf(z.reshape(-1))  for z in rt[:oom]]) # orig wbf  
#cols = np.array([fp.bwr(z.reshape(-1)) / 5 for z in Xt[:oom]]) # Xt bwr  (/5 for ds)
cols = np.array([fp.bwr(z.reshape(-1))  for z in rt[:oom]]) # orig bwr

bwrr = np.array([fp.bwr(z.reshape(-1))  for z in rt[:oom]])
bwrx = np.array([fp.bwr(z.reshape(-1))  for z in Xt[:oom]])

X = tsne.tsne[:,0]
Y = tsne.tsne[:,1]

unique_markers = set(Ym)  # or yo can use: np.unique(m)

    
import matplotlib.pyplot as plt
#plt.figure();plt.scatter(tsne.tsne[:,0], tsne.tsne[:,1], c=sizes, marker=Ym)

plt.figure()
for um in unique_markers:
    mask = Ym == um
    plt.scatter(X[mask], Y[mask], c=cols[mask], marker=um)
