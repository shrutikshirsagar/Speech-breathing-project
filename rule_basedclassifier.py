#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:28:21 2020

@author: shruti
"""

from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal

x = pd.read_csv('/media/shruti/Data/Breathing_project/CRD_interspeech_new.csv', header=0)
x = np.array(x)
#x_test = x_test[:, 1:]
print(x.shape)
X1 = x[:, 1]
X1= X1.squeeze()
X1 = X1[:, None]
print(X1.shape[0])
X2 = x[:, 9]
X2= X2.squeeze()
X2 = X2[:, None]
print(X2.shape)
X3 = x[:, 13]
X3= X3.squeeze()
X3 = X3[:, None]
print(X3.shape)
X4 = x[:, 0]
X4= X4.squeeze()
X4 = X4[:, None]
print(X4.shape)
Y = x[:,-1]
Y = Y.squeeze()
print(Y.shape)

Y_test =[]

for i in range(x.shape[0]):
    #print(X[i])
    if X1[i]>0.55 and X2[i]>0.25:
    # if X1[i]>0.45 and X2[i]>0.25 and  X3[i]<2.0:
    # if  X3[i]<1.7:
    #if X[i]>0.45:  
        Y_test.append(0)
    else:
        Y_test.append(1)
    
print(Y_test)
a = np.sum(Y_test)
print(a)

Y_test = np.asarray(Y_test)
acc=100*sum((Y_test-Y)==0)/len(Y)
print('accuracy', acc) 