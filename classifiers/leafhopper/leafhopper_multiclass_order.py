import pandas as pd 
import fpmodules as fp
import fpmodules.tools as fk

from fplearn.processing import  mlready_data, compile_process_segmented_data
from fplearn.run import run_model, evaluate_model
from fplearn.tsne import TSNEplot

import matplotlib.pyplot as plt
import numpy as np


event_folder = r'C:\Users\Server1\EventCache\Events'

data_length = 1000
session_groups = [
    [827],  
    [467, 476, 513, 938, 1073, 865], 
    [1095],
    [750, 850, 879, 717, 724, 753, 828],
    [777, 816, 839, 873, 954, 363], 
    [830, 469, 567, 734, 449, 450, 451, 452]   
]

classes_short = [
    'S. titanus_(Hemiptera)',
    'Hemiptera_(Rest)',
    'Thysanoptera',
    'Hymenoptera',
    'Coleoptera',
    'Diptera'  
]


max_samples = [
    [16000], 
    [1500, 1500, 3000, 3000, 3000, 3000],
    [16000],
    [3000, 3000, 3000, 2000, 2000, 2000, 2000],
    [2000, 2000, 3000, 3000, 3000, 3000],
    [3000, 3000, 3000, 3000, 1000, 1000, 1000, 1000]
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
#    ftfilt_low=[{52:0}, {53:0}], 
#    ftfilt_high=[{52:400}, {53:0.9}],
    split_channels=True, 
    data_length=data_length, 
    verbose=0, 
    download=True,
    thresh_low=0.7,
    thresh_high=1
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

batch_size=300
epochs=40
stop_patience=12
learning_rate=0.001
label_smoothing=0.3

model, params = run_model(
    Xt, 
    Xv, 
    Yt, 
    Yv,
    class_weights=cw,
    batch_size=batch_size,  
    epochs=epochs, 
    stop_patience=stop_patience, 
    label_smoothing=label_smoothing,
    learning_rate=learning_rate,
    arch='STANDARD'
)

#model.save(r'C:\Users\Server1\ML_models\leafhopper_multiclass_order-arch_stand')

#import keras
#model = keras.models.load_model(r'C:\Users\Server1\ML_models\leafhopper_multiclass_order-arch_stand')


import json
import pickle
base_path = "C:/Users/Server1/ML_models/leafhopper_multiclass_order_arch_stand_970/"
modelname = "leafhopper_multiclass_order_arch_stand_970"
model.save(base_path + modelname + ".h5", save_format='h5')

params_json = json.dumps(str(params))
with open(base_path + modelname + "_params.json", "w") as fh:
    fh.write(params_json)

with open(
        base_path + 'history',
        'wb'
        ) as file_pi:
    pickle.dump(model.history.history, file_pi)




np.save('my_history.npy',model.history.history)
history=np.load('my_history.npy',allow_pickle='TRUE').item()




acc_dict, \
matrix, \
score = evaluate_model(
    model, 
    model.history.history, 
    Xe.astype(float), 
    Ye, 
    classes_short
)


from tensorflow.keras.models import load_model
model = load_model(base_path + modelname + ".h5")

params_json = open (base_path + modelname + "_params.json", "r") 
params = json.loads(params_json.read())
history = pickle.load(
    open(
        'C:/Users/Server1/ML_models/leafhopper_multiclass_order-arch_stand/history',
        "rb"
        )
    )

# =============================================================================
# 
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense
# from tensorflow.python.keras.layers import deserialize, serialize
# from tensorflow.python.keras.saving import saving_utils
# 
# 
# def unpack(model, training_config, weights):
#     restored_model = deserialize(model)
#     if training_config is not None:
#         restored_model.compile(
#             **saving_utils.compile_args_from_training_config(
#                 training_config
#             )
#         )
#     restored_model.set_weights(weights)
#     return restored_model
# 
# # Hotfix function
# def make_keras_picklable():
# 
#     def __reduce__(self):
#         model_metadata = saving_utils.model_metadata(self)
#         training_config = model_metadata.get("training_config", None)
#         model = serialize(self)
#         weights = self.get_weights()
#         return (unpack, (model, training_config, weights))
# 
#     cls = Model
#     cls.__reduce__ = __reduce__
# 
# # Run the function
# make_keras_picklable()
# 
# import pickle
# 
# with open(r'C:\Users\Server1\ML_models\leafhopper_multiclass_order-arch_stand\params.pickle', 'wb') as handle:
#     pickle.dump(params, handle)
# 
# file = open(r'C:\Users\Server1\ML_models\leafhopper_multiclass_order-arch_stand\params.pickle', 'rb')    
# params = pickle.load(file)    
# =============================================================================

#%%
# tsne

oom = 3000

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


#%% Test on field sessions

from keras.utils import np_utils  # we need to get rid of this

sessid = 1052
maxi = 2000
ylen = 2

exdata,\
exlabels, \
_, \
_, \
exmids, \
_, \
_ = compile_process_segmented_data(
    [sessid],
    base_path,
    maxi, 
    data_length=data_length,
    num_channels=1)


exlabels = [1]* len(exlabels)
Xex = exdata.reshape([-1, data_length, 1]).astype(np.float32)
Yex = np_utils.to_categorical(exlabels, ylen).astype(np.float32)

Yexpred = model.predict(Xex)
from collections import Counter
print(Counter(Yexpred.argmax(axis=1)))

#exeval = evaluate_model(model, None, Xex, Yex, classes_short, lossplot=False)
# INERT SESSION 612 GETS TO BE CLASSIFIED AS SPANISH BEES...


#%% test what THE MACHINE LEARNS
import fpmodules as fp
#%% what the ml learns
wbfs = np.array([fp.wbf(r[0]) for r in rt])[:oom]
locs = np.where((wbfs > 70) & (wbfs < 200))[0]
fcol = wbfs[locs]
print(len(fcol))
plt.figure()
plt.scatter(tsne.tsne[locs,0], tsne.tsne[locs,1], c=fcol, alpha=.8)
plt.colorbar()
plt.title('wbf')
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
plt.xticks([])
plt.yticks([])

#%% length
lengths = np.array([r.shape[1] / 20831 for r in rt])
locs = np.where((lengths > 0.1) & (lengths < 2.5))[0]
fcol = lengths[locs]
plt.figure()
plt.scatter(tsne.tsne[:,0], tsne.tsne[:,1], c=fcol[:oom], alpha=.5)
plt.colorbar()
plt.title('t-SNE - event length (s)')
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
plt.xticks([])
plt.yticks([])
#%% bwr & friends
bwrs = np.array([fp.bsr(r[0]) for r in rt[:oom]])
locs = np.where((bwrs > 0) & (bwrs < 500))[0]
fcol = bwrs[locs]
print(len(fcol))
plt.figure()
plt.scatter(tsne.tsne[locs,0], tsne.tsne[locs,1], c=fcol, alpha=.8)
plt.colorbar()
plt.title('BW ratio')
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
plt.xticks([])
plt.yticks([])

#%%
sized = np.array([np.log(-np.min(r[0])) for r in rt])[:oom]
locs = np.where((sized > 0) & (sized < 1000))[0]
fcol = sized[locs]
print(len(fcol))
plt.figure()
plt.scatter(tsne.tsne[locs,0], tsne.tsne[locs,1], c=fcol, alpha=.8)
plt.colorbar()
plt.title('median')
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
plt.xticks([])
plt.yticks([])




