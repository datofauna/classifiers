
import pandas as pd
import sys
sys.path.append('/home/frth/notebooks')
import keras.models

import fpmodules as fp
import fpmodules.tools as fk

feature_id = list(fk.get_feature_ids().values())
#from helpers import * 
import sys
from fplearn.processing import  mlready_data, compile_process_segmented_data
from fplearn.run import run_model, evaluate_model
from fplearn.tsne import TSNEplot

import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('/home/frth/notebooks/syngenta/model/multi_class_1000_w_1014_half_network_nb')

#sessions = [806,807,808,809,810,811]
sessions = [811]
insect_pb = pd.DataFrame()
for s in sessions:
    insects = fp.get_insects(sessionid=s,all_segments=True,fractional_classifier=None,noise_classifier=31)

    field_data = \
        compile_process_segmented_data([s], '/home/frth/EventCache/Events',split_channels=True, data_length=1000, num_channels=1, verbose=1)
    
    field_data = np.expand_dims(field_data[0], axis=2)
    insects = insects.sort_values(['MeasurementId', 'SegmentId'])    
    insects['Pollen beetle'] = 0.
    for i in range(0, len(field_data), 2000):

        if (i+2000 < len(field_data)):
            print(field_data[i:i+2000].shape)
            prediction = model.predict(field_data[i:i+2000])


            print(prediction.shape)
            print(len(insects))
            insects['Pollen beetle'][i:i+2000] = prediction[:,0]
            if len(insect_pb) == 0:
                insect_pb = insects[insects['Pollen beetle'] >= 0.7]
            else:
                insect_pb = insect_pb.append(insects[insects['Pollen beetle'] >= 0.7])

insect_pb.to_pickle('/home/frth/notebooks/syngenta/pollen_beetle_811.pkl')
