#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:29:14 2020

@author: shruti
"""

from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal


# label = pd.read_csv('/home/shruti/Downloads/Master_list.csv', header=0)
# l = np.array(label)
# print(l.shape[0])
# l1 = l[:,3]
# l2 = l[:,9]

# for i in range(x.shape[0]):
#     #print(X[i])
#     if l1[i]>0.45 or l2[i]==Day shift:


import pandas as pd
output_fin=np.empty((0,9))
output_cols=['participant_id','Shift Info','sex', ' talk', 'total', 'notalk', 'talk_to_total', 'notalk_to_total', 'talk_notalk' ]
#load master list data
info_csv='/home/shruti/Downloads/Master_list.csv'
df_info=pd.read_csv(info_csv)[['participant_id','Shift Info','sex']]

#iterate over dataframes
for idx,row in df_info.iterrows():
	#get a subject , its shift and sex information
    sub_nm,shift,sex=str(row['participant_id']),str(row['Shift Info']),str(row['sex'])
    print(sub_nm,shift,sex)
    file_path='/media/shruti/Data/Breathing_project/Tiles/br_fts/'+sub_nm+'.csv'
    print(file_path)
    try:
        df_br=pd.read_csv(file_path)
    
        x = pd.read_csv(file_path)
        x = np.array(x)
        x = x[:, 1:]
        print('x_shape', x.shape[0])
        X = x[:, 2]
        X = X.squeeze()
        X = X[:, None]
        #print(X.shape[0])
        X2 = x[:, -1]
        X2= X2.squeeze()
        X2 = X2[:, None]
        #print(X2.shape)
        Y_test =[]
        
        for i in range(x.shape[0]):
        
            if X[i]>0.45 or X2[i]>0.25:
                Y_test.append(0)
            else:
                Y_test.append(1)
        
        #print(Y_test)
        talk = np.sum(Y_test)
        print(talk)
        total = x.shape[0]
        notalk = total - talk
        talk_to_total = talk/total
        notalk_to_total = notalk/total
        talk_notalk= talk/notalk
        out_vec=np.hstack((sub_nm,shift,sex,talk,total,notalk,talk_to_total,notalk_to_total,talk_notalk))
        output_fin=np.vstack((output_fin,out_vec))
    except:
        continue
df=pd.DataFrame(output_fin,columns=output_cols)
df.to_csv('/media/shruti/Data/Breathing_project/day_night_analysis_new_new.csv',index=None)