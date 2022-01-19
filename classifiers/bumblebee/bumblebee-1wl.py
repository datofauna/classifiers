# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:47:58 2021

@author: Server1
"""
import fpmodules as fp
import numpy as np
import os
from fplearn.processing import process_insect_data_wmax
from fplearn.transform import split_scout_segments
import random
from fpmodules.tools.common import to_list
from fpmodules.tools.constants import DEF_NOISE_CLASSIFIERID



def split_scout_channels(data_list, *other_lists, samples_n=None):
    """
    Splits segmented scout data into individual channels. Also does the same
    parallelly for other lists passed through, with the option of random 
    sampling.

    Parameters
    ----------
    data_list : list
        List of segmented scout data to split.
    *other_lists : lists
        Further lists (like measurement IDs, filenames etc) to split in parallel.
    samples_n : int, optional
        If given, selects a random sample of this number. The default is None.

    Returns
    -------
    fulldata : tuple
        tuple containing the split data lists.

    """
    # Note that random state is not possible to be kept here
    data970 = [d[0,:].reshape([1, -1]) for d in data_list]
    data810 = [d[1,:].reshape([1, -1]) for d in data_list]
    # All this shuld be data_dict
    data_conc = data970 + data810
    proc_other_lists = []
    for extra_list in other_lists:
        proc_other_lists.append(extra_list + extra_list)
    wave_list = [970] * len(data_list) + [810] * len(data_list)
    if samples_n:
        fulldata = tuple(zip(*random.sample(
            list(zip(data_conc, wave_list, *proc_other_lists)), samples_n)))
    else:
        fulldata = (data_conc, wave_list, *proc_other_lists)
    return fulldata 


def read_data_segments(sessionid:int, download_path:str,
                       max_samples=None, verbose=2,
                       noise_classifier=DEF_NOISE_CLASSIFIERID, threshold=.7):
    """
    Reads each segment as a datapoint for a given session. Downloads events if 
    necessary. Segments are read that pass a certain classification threshold.

    Parameters
    ----------
    sessionid : int
        Session ID.
    download_path : str
        Path to download events to.
    max_samples : int (or None), optional
        Limits the samples to be read. The default is None.
    verbose : 0, 1 or 2, optional
        Verbosity level. The default is 2.

    Returns
    -------
    data : list
        List of np.arrays containing the event data.
    file : list
        List of strings containing the file names.
    mid : list
        List of Measurement IDs.
    seg : list
        List of Segment IDs.

    """
    if verbose > 0:
        print(f"*** Collecting {sessionid}, max samples: {max_samples} ...")
    if verbose > 1:
        print("-- Fetching insects ...")
    insects = fp.get_insects(sessionid=sessionid, noise_classifier=noise_classifier, # add to 31
                             threshold=threshold, # how will it react to noise?
                             fractional_classifier=None, all_segments=True)
    if int(max_samples or len(insects)) < len(insects):
        insects = insects.sample(n=max_samples, random_state=42)
        # Randomizing does not take the segments into account
    
    insectlist = insects['MeasurementId'].unique().tolist()
    if verbose > 1:
        print(f"-- {len(insectlist)} unique events ...")
    evlist = fp.EventList(insectlist, verbose=0,
            download_path=os.path.join(download_path, str(sessionid))).fill()
    data, file, mid, seg = split_scout_segments(evlist, insects)
    if verbose > 1:
        print(f"-- {len(data)} segment data points")
    data, file, mid, seg = clean_dataset(data, file, mid, seg, 
                                         verbose=verbose)
    return data, file, mid, seg


def compile_segmented_dataset(sesslist:list, download_path:str, max_samples=None,
                              split_channels=True, 
                              verbose=2):
    """
    Creates a set of data points by taking each segment of events in the given
    list of sessions that passes a threshold.

    Parameters
    ----------
    sesslist : list
        List of session IDs.
    download_path : str
        Path to download events to.
    max_samples : int or None or list thereof, optional
        Limits the samples to be read. The default is None.
    verbose : 0, 1 or 2, optional
        Verbosity level. The default is 2.
    split_channels : bool, optional
        If True, returns 1XN matrices, each channel is taken individually.
        The default is True.


    Returns
    -------
    data : list
        List of np.arrays containing the event data.
    file : list
        List of strings containing the file names.
    mid : list
        List of Measurement IDs.
    seg : list
        List of Segment IDs.
    wave : list
        List of Wavelength IDs.

    """
    sesslist = to_list(sesslist)
    if isinstance(max_samples, int) or max_samples is None:
        max_samples = [max_samples] * len(sesslist)
        
    data, file, mid, seg, wave = [], [], [], [], []         
    for sid, maxi in zip(sesslist, max_samples):        
        fmaxi = None if maxi is None else maxi // 2 
        data_, file_, mid_, seg_ = read_data_segments(sid, download_path, fmaxi,
                                                  verbose=verbose)
        wave_ = []
        if split_channels: # params for split_
            data_, wave_, file_, mid_, seg_ =\
                split_scout_channels(data_, file_, mid_, seg_)
            if verbose:
                print(f"-- {len(data_)} channel data points")
        if verbose > 1: print("\n")
        data += data_
        file += file_
        mid += mid_
        seg += seg_
        wave += wave_
    return data, file, mid, seg, wave

def clean_dataset(dataset, *other_datasets, verbose=1):
    """Removes empty data points paralelly from all datasets"""
    ix = [i for i, x in enumerate(dataset) if len(x) > 1]
    cleaned_dataset = [dataset[i] for i in ix]
    processed_others = []
    for other in other_datasets:
        processed = [other[i] for i in ix]
        processed_others.append(processed)
    if verbose:
        num_cleaned = len(dataset) - len(ix)
        print(f"! *** {num_cleaned} events removed")
    full = (cleaned_dataset, *processed_others)
    return full
    
            

def compile_process_segmented_data(grouplist, download_path, 
                                   max_samples=None, split_channels=True, 
                                   verbose=2, **proc_kw):
    """
    Compiles and processes data for given list of sessions by segment.

    Parameters
    ----------
    grouplist : list of lists
        Nested list of session IDs.
    download_path : str
        Path to download events to.
    max_samples : int or None or list (of lists) thereof, optional
        Limits the samples to be read. The default is None.
    verbose : 0, 1 or 2, optional
        Verbosity level. The default is 2.
    split_channels : bool, optional
        If True, returns 1XN matrices, each channel is taken individually.
        The default is True.
    **proc_kw : Keyword arguments
        Keyword arguments for the data processing.

    Returns
    -------
    data : np.array
        np.array containing the segmented and processed data.
    labels : list
        List containing assigned label values.
    file : list
        List of strings containing the file names.
    raw: list
        List of np.arrays with the unprocessed data.
    mid : list
        List of Measurement IDs.
    seg : list
        List of Segment IDs.
    wave : list
        List of Wavelength IDs.

    """
    raw, file, mid, seg, wave, data, labels = [], [], [], [], [], [], []
    if isinstance(max_samples, int) or max_samples is None:
        max_samples = [max_samples] * len(grouplist)
        
    for label, (group, maxi) in enumerate(zip(grouplist, max_samples)):
        if verbose: print(f"*** LABEL: {label} PROCESSING ***")
        raw_, file_, mid_, seg_, wave_ = \
            compile_segmented_dataset(group, download_path, 
                                      maxi, split_channels=split_channels,
                                      verbose=verbose)
        if verbose: print("Processing data ...")
        data_ = [process_insect_data_wmax(i, **proc_kw) for i in raw_]
        data += data_
        file += file_
        mid += mid_
        raw += raw_
        seg += seg_
        wave += wave_
        labels += [label] * len(data_)
    return np.concatenate(data), labels, file, raw, mid, seg, wave
        

# grouplist = [[499, 531], 37]
# split_channels=True
# max_samples=[[4, 3], 35000]
# verbose = 2
# download_path="Y:/fpCache/ML/Events/"
# alld = compile_process_segmented_data(grouplist, download_path, max_samples,
#                                       split_channels=True, data_length=500,
#                                       num_channels=1)

#%%
base_path = "Y:/fpCache/ML/Events/"
classes_short = ['Not bumblebee', 'Bumblebee']

# grouplist = [686]
# maxlist = [15000]
# alldt = compile_process_segmented_data(grouplist, base_path, maxlist,
#                                        split_channels=True, data_length=500,
#                                        num_channels=1)

#%%
from fplearn.processing import mlready_data
from fplearn.run import run_model, evaluate_model

grouplist = [[499, 531, 586, 674, 691, 717, # other hymenoptera
             23, 37, 122, 123, #other osr
             567, 341, 367, 168, 171, 527, # flies and midges
             537, 206, 456, 331, 464, 162, 716, 409, # diversity quota
             686, # honeybees
             687], # field honeybees
            [461, 462, 675, 266]] # bumblebees

maxlist = [[800, 800, 800, 800, 800, 800,
            500, 500, 500, 500,
            500, 1000, 500, 500, 500, 800,
            300, 300, 300, 300, 300, 300, 300, 300,
            5000,
            500], 
           8000]

data_length = 1000

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
                          batch_size=200, epochs=75, stop_patience=10)

evals = evaluate_model(model, model.history.history, Xe, Ye, classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
#%% Test on field sessions
from keras.utils import np_utils  # we need to get rid of this

sessid = 5
maxi = 1000
ylen = 2



exdata, exlabels, _, _, _, _, _ = \
    compile_process_segmented_data([sessid], base_path, maxi, 
                                         data_length=data_length, num_channels=1)
Xex = exdata.reshape([-1, data_length, 1])
Yex = np_utils.to_categorical(exlabels, ylen)

exeval = evaluate_model(model, None, Xex, Yex, classes_short, lossplot=False)

#%% tsne-evals
from fplearn.tsne import TSNEplot
import pandas as pd
oom = 4000
tsne = TSNEplot(Xex[:oom], Yex[:oom], model, classes=classes_short)
tsne.fit()
#tsne.plot()

def custom_onpick(scobj):
    ix = scobj.index
    title = f"""({ix}) {scobj.mids[ix]}: {scobj.originals[ix]} -> {str(scobj.predictions[ix])}"""
    print(title)
    scobj.axes[1].set_title(title)
    
sc = ScatterPicker(tsne.tsne[:,0], tsne.tsne[:,1], [d.T for d in Xex[:oom]])
# sc = fp.ScatterPicker(tsne.tsne[:,0], tsne.tsne[:,1], [d.T for d in Xt[:oom]], 
#                       custom_onpick=custom_onpick, keep_default=False)
eval_classes = np.argmax(Yex, axis=1)[:oom]
cols = ["red" if d == 0 else "green" for d in eval_classes ]
sc.scatter_kw = {"c":cols, 'alpha': .5}
sc.rescatter_kw = sc.scatter_kw
sc.plot_kw = {"color": "black"}

sc.originals = Yt[:oom][:,1] # Specific for noise (binary) classifier
sc.predictions = model.predict(Xex[:oom])[:,1].round(2)
sc.mids = mt[:oom]

sc.display()


#%% I/O
# modelname = "bumblebee-bs1-fl"
# model.save(base_path + modelname + ".h5", save_format='h5')

# from tensorflow.keras.models import load_model
# model = load_model(base_path + modelname + ".h5")

#%% test sensitivities
import matplotlib.pyplot as plt

maxi = 5000
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

results.to_csv(base_path + "bumblebee_1wl_sensitivities.csv")