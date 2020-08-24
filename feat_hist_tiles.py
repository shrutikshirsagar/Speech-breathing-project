#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:12:38 2020

@author: shruti
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

info_csv='/home/shruti/Downloads/Master_list.csv'
df_info=pd.read_csv(info_csv)[['participant_id','Shift Info','sex']]

#iterate over dataframes
for idx,row in df_info.iterrows():
	#get a subject , its shift and sex information
    sub_nm,shift,sex=str(row['participant_id']),str(row['Shift Info']),str(row['sex'])
   
    file_path='/media/shruti/Data/Breathing_project/Tiles/br_fts/'+sub_nm+'.csv'
    path = '/media/shruti/Data/Breathing_project/hist_plots/'+sub_nm + '/'
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        x=pd.read_csv(file_path)
    
        x = np.array(x)

        print('shape', x.shape)
        an = np.arange(0, 14, 1)
        for i in an:
        
            X = x[:, i]
            X = np.array(X)
            print('shape of data', X.shape)
            plt.figure()
            sns.kdeplot(X, label="features_hist")
            
           
            plt.legend();
            plt.savefig(path + str(i)+ '.png')
            plt.clf()
          

        
      
        
       
    except:
        continue
