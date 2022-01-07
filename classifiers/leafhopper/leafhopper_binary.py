import pandas as pd 
import fpmodules as fp
import fpmodules.tools as fk

from fplearn.processing import  mlready_data, compile_process_segmented_data
from fplearn.run import run_model, evaluate_model
from fplearn.tsne import TSNEplot

import matplotlib.pyplot as plt
import numpy as np


event_folder = r'C:\Users\StylianosNikolaou\EventCache\Events'

data_length = 1000
session_groups = [
    [
     822, 
     827
     ], 
    [
     363, 
     476, 
     717, 
     750, 
     816, 
     828, 
     830, 
     839, 
     873, 
     879, 
     1224
     ] 
]

max_samples = [
    5000, 
    2000
]    


classes_short = [
    'Leafhopper', 
    'Non-Leafhopper'
]
 
 
data, \
labels, \
files, \
raw, \
mid, \
seg, \
wave = compile_process_segmented_data(
    session_groups, 
    event_folder,
    max_samples, 
    split_channels=True, 
    data_length=data_length, 
    verbose=0, 
    download=True
)



(
 Xt, Xv, Xe, 
 Yt, Yv, Ye,
 ft, fv, fe, 
 rt, rv, re, 
 mt, mv, me, 
 st, sv, se, 
 wt, wv, we
 ), \
 cw = mlready_data(
    data, 
    labels, 
    files, 
    raw, 
    mid, 
    seg, 
    wave
)

Xt = Xt.reshape([-1, data_length, 1])
Xv = Xv.reshape([-1, data_length, 1])
Xe = Xe.reshape([-1, data_length, 1])

batch_size=100
epochs=25
stop_patience=12
learning_rate=0.001

model, params = run_model(
    Xt, 
    Xv, 
    Yt, 
    Yv, 
    class_weights=cw,
    batch_size=batch_size, 
    epochs=epochs, 
    stop_patience=stop_patience, 
    learning_rate=learning_rate
)

model.save(r'C:\Users\StylianosNikolaou\models\leafhopper_non-leafhopper')

# model = keras.models.load_model(r'C:\Users\StylianosNikolaou\models\leafhopper_non-leafhopper')

acc_dict, \
matrix, \
score = evaluate_model(
    model, 
    model.history.history, 
    Xe.astype(float), 
    Ye, 
    classes_short
)



oom = 1000

tsne = TSNEplot(
    Xt[:oom], 
    Yt[:oom], 
    model, 
    classes=classes_short
)
tsne.fit()
tsne.plot()

plt.gcf().set_size_inches(6., 4.)
tsne_fig = plt.gcf()