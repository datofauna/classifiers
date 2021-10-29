import sys
sys.path.append('/home/frth/notebooks')

import fpmodules as fp
import fpmodules.tools as fk

feature_id = list(fk.get_feature_ids().values())
import sys
from fplearn.processing import  mlready_data, compile_process_segmented_data
from fplearn.run import run_model, evaluate_model
from fplearn.tsne import TSNEplot

import matplotlib.pyplot as plt
import numpy as np

#%%
species = {
    'Pollen_beetle/Danes': 816,
    'Pollen_beetle/Swiss': 777,
    'Drosophilidae/Swiss': 794,
    'Honeybee': 805,
    'Greenhouse rove beetle': 409,
    'Cabbage steam weevil': 773,
    'Mealy cabbage aphid': 962,
    'Aphis gossipii': 988,
    'Lygocoris pabulinus': 1118,
    'Aphis fabae': 513,
    'Cabbage seed weevil': 873,
    'Codling moth': 588,
    'Pod midge': 985,
    'Bumblebee': 462,
    'Parasitic wasp': 750,
    'Green lacewing': 646,
    'Tersilochus heterocerus': 879,
    'House fly': 469,
    'Common green bottle fly': 452,
    'Aphid gall midge': 567,
    'Migrant hoverfly': 478
}
labels = list(species.keys())

#%%
#for l in labels:
#    path='/home/' + user + '/EventCache/Events/' + str(species[l]['session'])
#    fp.download_events(id_list=species[l]['meas_id'],multi=True, path=path)
#%%
[species[s]['session'] for s in species]
session_groups = [816,777]#[[816, 777, 1014],  794, 805, 409, 773, 962, 988, 1118, 513, 873, 588, 985, 462, 750, 646, 879, 469, 452, 567, 478]
maxlist = [10,10]#[[100, 100, 100]] + [100]*19
startdateid = [20210101, None]
#%%
#for s in species:
#    print(len(fp.get_insects(sessionid=species[s]['session'])))
#%%
data, labels, files, raw, mid, seg, wave = \
    compile_process_segmented_data(session_groups, '/home/frth/EventCache/Events',
                                   maxlist, split_channels=True, data_length=1000, verbose=0, startdateid=startdateid)
#%%
data
#%%
data_length = 1000
(Xt, Xv, Xe, Yt, Yv, Ye, \
 ft, fv, fe, rt, rv, re, mt, mv, me, st, sv, se, wt, wv, we), cw = \
    mlready_data(data, labels, files, raw, mid, seg, wave)
Xt = Xt.reshape([-1, data_length, 1])
Xv = Xv.reshape([-1, data_length, 1])
Xe = Xe.reshape([-1, data_length, 1])
#%%
#classes_short = ['Pollen_beetle', 'Drosophilidae', 'Honeybee', 'Greenhouse rove beetle', 'Cabbage steam weevil'] # list(species.keys())
#%%
model, params = run_model(Xt, Xv, Yt, Yv, class_weights=cw,
                          batch_size=200, epochs=300, stop_patience=10, learning_rate=0.0005)
model.save('/home/frth/notebooks/syngenta/model/multi_class_1000_w_1014_2')

#model = keras.models.load_model('/home/frth/notebooks/syngenta/model')
#%%
classes_short = ['Pollen beetles'] + list(species.keys())[2:]
plt.figure(figsize=(15,15))
evals = evaluate_model(model, model.history.history, Xe.astype(float), Ye, classes_short)
#%%
params.update({"sessions": str(session_groups),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%%
from fplearn.tsne import TSNEplot
oom = 1000
plt.figure(figsize=(12,8))
tsne = TSNEplot(Xt[:oom], Yt[:oom], model, classes=classes_short)
tsne.fit()
tsne.plot()
plt.gca().set_xlabel('test')
plt.gcf().set_size_inches(18.5, 10.5)
tsne_fig = plt.gcf()