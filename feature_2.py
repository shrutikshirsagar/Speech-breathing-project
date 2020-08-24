#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:21:50 2020

@author: shruti
"""

from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 

#put breathing data here as a numpy array
import pandas as pd
from scipy import signal

import scipy.io as sio

from scipy.signal import periodogram, welch
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
import librosa
from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal
from pyentrp import entropy as ent
label = 'upper_belt'
df = pd.read_csv('/media/shruti/Data/Breathing_project/Interspeech_Breathing/lab/labels.csv', delimiter=',')
print(df.shape)
breathing=np.empty((6000,0))
#y_devel = df[label][df['filename'].str.startswith('devel')].values
#print('deve', y_devel.shape)
num = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14','15']
from mne_features.univariate import compute_spect_slope
output_cols = ['bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','p1_p2','p1_p3','kurt','sknew','flatness','peak_f','ratio_25', 'ratio_50','ratio_150','perm_ent', 'labels']
def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.log(X + 1e-20)

    vtf = np.exp(X.mean(axis=0, keepdims=True)) / norm

    return (vtf)

feat = np.empty((0, 14))
for i in num:
    print(i)
    df2 = df.loc[df['filename'] == 'devel_'+ i+ '.wav']
    print(df2.shape)

    df3 = df2.values

    t = df3[:,1]
    t = t[:, None]
    t = np.concatenate(t).astype(None)

    t1 = np.asarray(t)
    print(t1.shape)
    t1 = t1[:, None]
    print(t1.shape)
    s = df3[:,2]
    s = s[:, None]
    s = np.concatenate(s).astype(None)

    s1 = np.asarray(s)
    s1 = s1[:, None]
    print(s1.shape)


    br= s1
    br=(br-np.mean(br))/np.std(br)
    br = signal.resample(br, len(br)*4)

    br = signal.decimate(br, 5, axis =0)

    br = signal.decimate(br, 5, axis =0)


    fs=4  #sampling frequency
    w_size =  20 * fs   # window size in seconds
    w_shift = 15 * fs  # window overlap

    #compute short time fourier transform
    rfft_spect_h = ama.strfft_spectrogram(br, fs, w_size, w_shift, win_function = 'hamming' )
    power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
    print(rfft_spect_h['power_spectrogram'].shape[0])
    br_epoch,_,_=ama.epoching(br, w_size, 5*fs)
    br_epoch = br_epoch.squeeze()

   
    for ix in range(0,rfft_spect_h['power_spectrogram'].shape[0]):
        psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
        psd=psd/np.sum(psd)
        
        a = rfft_spect_h['freq_axis']
       
        b = np.where(a<0.25) 
        feat_1 = psd[b]
        feat_1 = np.sum(feat_1)
        #print(feat_1)
      
        c = np.where((a >0.25) & (a<0.5))
        feat_2 = psd[c]
        feat_2 = np.sum(feat_2)
        #print(feat_2)
        
        d = np.where((a >0.5) & (a<1.5))
        feat_3 = psd[d]
        feat_3 = np.sum(feat_3)
        #print(feat_3)
        
        feat_4 = -np.sum(psd*np.log(psd))
        #print(feat_4)
        
        
        
        
        
        peaks, properties = find_peaks(psd)
        
        a1 = psd[peaks]
        #plt.plot(psd)

        a1=np.sort(a1)
        first, second, third=a1[-1], a1[-2], a1[-3]
     
        #print(first, second, third)
        feat_5=first/second
        #print(feat_5)
        feat_6=first/third
        #print(feat_6)
        feat_7 = kurtosis(psd)
        #print(feat_7)
        feat_8 = skew(psd) 
        #print(feat_8)
       
        #print(feat_9)
        
        
        
        feat_10 = FeatureSpectralFlatness(psd, fs)
        #print(feat_10)
        
        feat_11 = a[np.argmax(psd)]
    
        #in band vs out of band ratio for feat 1
        feat_12=feat_1/(feat_2+feat_3)
    
        #in band vs out of band ratio for feat 1
        feat_13=feat_2/(feat_1+feat_3)
    
        #in band vs out of band ratio for feat 1
        feat_14=feat_3/(feat_1+feat_2)
    
        #get timeseries entropy (PE)
        lag=1
        x_ = br_epoch[:, ix]
        print(x_.shape)
        feat_15=ent.permutation_entropy(x_,3,lag)
        feat_c=np.hstack((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_10, feat_11,feat_12,feat_13,feat_14,feat_15))
        feat=np.vstack((feat,feat_c))
n= feat.shape[0]
print(n)
X0 = np.ones((n,1))
print(X0.shape)
feat = np.hstack((feat,X0))
print(feat.shape)

df=pd.DataFrame(feat,columns=output_cols)
df.to_csv('/media/shruti/Data/Breathing_project/breathing_talk_devel_new.csv',index=None)
			
