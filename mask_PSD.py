#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:37:02 2020

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

import os
import pandas as pd
#output_fin=np.empty((0,12))
output_cols=['bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','p1_p2','p1_p3','kurt','sknew','flatness','peak_f','ratio_25', 'ratio_50','ratio_150','perm_ent', 'filename', 'labels']
#load master list data
info_csv='/home/shruti/Downloads/Master_list.csv'
df_info=pd.read_csv(info_csv)[['participant_id','Shift Info','sex']]
path = '/media/shruti/Data/Breathing_project/mask/mask_3_019/'
if not os.path.exists(path):
    os.makedirs(path)
#iterate over dataframes
for idx,row in df_info.iterrows():
	#get a subject , its shift and sex information
    sub_nm,shift,sex=str(row['participant_id']),str(row['Shift Info']),str(row['sex'])
    print(sub_nm,shift,sex)
    file_path='/media/shruti/Data/Breathing_project/Tiles/br_fts/'+sub_nm+'.csv'
    print('name',file_path)
   
    # df_br=pd.read_csv(file_path)
    try:
        x = pd.read_csv(file_path)
        cols=x.columns.values.tolist()
        print(cols)
    except:
        continue
    # x = np.array(x)
    # print(x.shape)
    X1 = x['bnd_pwr_25_50']
    X1 = np.array(X1)
    X1 = X1[:,None]
    print(X1.shape)
    X2 = x['peak_f']
    X2 = np.array(X2)
    X2 = X2[:,None]
    print(X2.shape)
    X3 = x[ 'entropy']
    X3 = np.array(X3)
    X3 = X3[:,None]
    print(X3.shape)
    X4 = x[ 'bnd_pwr_0_25']
    X4 = np.array(X4)
    X4 = X4[:,None]
    print(X4.shape)
    x_new = np.array(x)
    print(x_new.shape[0])
    # X = x['':, 1'']
    # X = X.squeeze()
    # X = X[:, None]
    # #print(X.shape[0])
    # X2 = x[:, -6]
    # X2= X2.squeeze()
    # X2 = X2[:, None]
    # X3 = x[:, 3]
    # X3 = X.squeeze()
    # X3 = X[:, None]
      
    Y_test =[]
    
    for i in range(x_new.shape[0]):
    
        # if X1[i]>0.45 and X2[i]>0.25 and X3[i]<1.7:
        #if X1[i]>0.45:
        #if X2[i]>0.25:
        # if X3[i]<1.7:
        if X1[i]>0.55 and X2[i]>0.25 and X4[i]<0.2:
            #if X2[i]>0.25:
            Y_test.append(0)
        else:
            Y_test.append(1)
        
        
    Y = np.asarray(Y_test)
    Y = Y[:, None]
    print('labels', Y.shape)
    out_vec=np.hstack((x_new, Y))
    print(out_vec.shape)
    df=pd.DataFrame(out_vec, columns=output_cols)
    df.to_csv(path+sub_nm+'.csv' ,index=None)

