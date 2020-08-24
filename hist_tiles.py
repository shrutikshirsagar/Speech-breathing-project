#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:14:01 2020

@author: shruti
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
sns.set(); np.random.seed(0)


path='/media/shruti/Data/Breathing_project/Tiles/br_fts/'
fnms=os.listdir(path)
cols=['entropy','flatness','peak_f']
for fnm in fnms:
   
    df=pd.read_csv(path+fnm)
    df=df[cols]
    for col in cols:
      
        arr=np.array(df[col])
        ax = sns.distplot(arr)
        plt.figure()
        plt.savefig('/media/shruti/Data/Breathing_project/hist_plots/' + col + '.png')
        plt.clf()