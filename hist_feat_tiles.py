#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:57:27 2020

@author: shruti
"""

from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from scipy import stats
x = pd.read_csv('//media/shruti/Data/Breathing_project/Tiles/br_fts/2a8f6d94-88d4-4d72-b304-782563903dca.csv', header=0)
x = np.array(x)

print(x.shape)
an = np.arange(1, 11, 1)
for i in an:

    X = x[:, i]
    X = np.array(X)
    print(X.shape)
    # # Y = x[:,-1]
    # # print(Y)
    plt.figure()
    # b = np.where(Y==0) 
    # c = X[b]
    # d = np.where(Y==1) 
    # c1 = X[d]
    sns.kdeplot(X, label="no talk")
    # sns.kdeplot(c1,  label="talk")
    
    plt.legend();
    plt.savefig('/media/shruti/Data/Breathing_project/hist_plots/'+'sub10_'+ str(i)+ '.png')
    plt.clf()
