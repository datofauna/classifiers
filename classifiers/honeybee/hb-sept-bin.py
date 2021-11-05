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
"""Methodology

- Find all labelled sessions where the median WBF is between 180 and 270
- That list goes in the list of sessionIDs here...
    
    select SS.Id, SS.SubjectCompositionId, ss.[Name], SI.[Order], SI.Family, SI.Genus, SI.species, COUNT(m.id) as NumEvents
    from [Session] SS join subjectinfo SI on ss.subjectcompositionid=si.subjectcompositionid 
    join Measurement m on m.SessionId=SS.Id
    where ss.Id in (28, 125, --248, 
    478, 484, 498, 499, 586, 629, 686, 706, 749, 799, 805, 817, 835, 836, 889, 901, 908, 909, 910, 920, 933, 985,
     996, 1046, 1051, 1052, 1054, 1055, 1059, 1065, 1082, 1107, 1111, 1112, 1113, 1114, 1115,
     1116, 1122, 1129, 1140, 1141, 1149, 1161,
     1162, 1163, 1164, 1165) and [Name] not like '%Tunnel %' and [Name] not like 'Flower Strip,%' and [Name] not like 'Tent %' and [Name] not like 'In between%' 
     and ss.SubjectCompositionId <>  --apismellifera
    
     group by SS.Id, SS.SubjectCompositionId, ss.[Name], SI.[Order], SI.Family, SI.Genus, SI.species
     order by SI.[order], SI.Family

- Disregard all things like "Tunnel" etc (non-pure samples)
- Disregard all Coleoptera and Hemiptera... probably errors
- Consider including background sessions (OSR=956, WhiteClover=1057)

"""


# classes_short = ['Not honeybee',# 'Bombus pascuorum', 
#                  'Honeybee']
# grouplist = [[749, 1072, 1046, 168, 171, 484, 478, 1139, 1141], 
#     [817,836,686,799,805,1052]
#              ]#, [527, 526]]#,498]
#     # bombus pascuorum, all hoverflies, osmia rufa
    
# grouplist = [
#     [1141, #leafcutter bee
#      749, #solitary bee osmia rufa (a.vaga has low wbf)
#      996, 1065, 499, 629, 586, #parasitoids (only high wbf)
#      1082, 1129, #bombus pascuorum-messy, bombus impatiens
#      498, 889, #Drosophila suzukii
#      478, 484, 1046, #Syrphidae
#      ], # musca domestica not included
#     [817, 836,686,799,805,1052]
#     ]


grouplist = [
      [1141, #leafcutter bee
      749, #solitary bee osmia rufa (a.vaga has low wbf)
      996, 1065, 499, 629, 586, #parasitoids (only high wbf)
      1082, 1129, #bombus pascuorum-messy, bombus impatiens
      498, 889, #Drosophila suzukii
      478, 484, 1046], #Syrphidae
       # musca domestica not included
      [817, 836,686,799,805,1052] # APIS MELLIFERA, all sbsp.
    ]
classes_short = ['Not honeybee', 'Honeybee']

    
data_length = 1000
#maxlist = [15000 // len(x) for x in grouplist]
#maxlist = [10000, 10000, 2000, 5000, 5000, 4000, 3000]
maxlist = [1000, 2500]

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
model_bin, params = run_model(Xt, Xv, Yt, Yv, class_weights=cw,
                          batch_size=500, epochs=50, stop_patience=12, learning_rate=0.0005)

evals = evaluate_model(model_bin, model_bin.history.history, Xe.astype(float), Ye, classes_short)
params.update({"sessions": str(grouplist),
               "test_accuracy": evals[-1][-1]})
params.pop('callbacks')
#%%
from fplearn.tsne import TSNEplot
oom = 3000
tsne = TSNEplot(Xt[:oom], Yt[:oom], model_bin, classes=classes_short)
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
