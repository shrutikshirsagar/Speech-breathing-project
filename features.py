"""Code to compute and generate the spectrogram for breathing using the amplitude modulation toolbox"""
from __future__ import division
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
from mne_features.univariate import compute_spect_slope
from pyentrp import entropy as ent
output_cols = ['bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','p1_p2','p1_p3','kurt','sknew','flatness','peak_f','ratio_25', 'ratio_50','ratio_150','perm_ent', 'labels']
#output_cols=['bndpwr_0.25','bndpwr_0.5','bndpwr_1.5','entropy','peak1_2','peak1_3', 'kurtosis', 'skew',  'flatness', 'max_freq', 'labels']
def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.log(X + 1e-20)

    vtf = np.exp(X.mean(axis=0, keepdims=True)) / norm

    return (vtf)
num = ['5', '7', '11', '13', '14', '18', '26',  '28', '34', '37','38', '39',  '42',  '48']
feat = np.empty((0, 14))
for i in num:
    br = sio.loadmat('/media/shruti/Data/Breathing_project/crd_biovad_data/CRD_new/br_sub_'+ i+ '.mat')
    #print(br.keys())
    s1 = br['br_fin']
    s1 = s1.squeeze()

    
    br= s1
    br=(br-np.mean(br))/np.std(br)
    fs=4  #sampling frequency
    w_size =  20 * fs   # window size in seconds
    w_shift = 15 * fs   # window overlap

    #compute short time fourier transform
    rfft_spect_h = ama.strfft_spectrogram(br, fs, w_size, w_shift, win_function = 'hamming' )
    power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
    #print(rfft_spect_h['power_spectrogram'].shape[0])
    br_epoch,_,_=ama.epoching(br, w_size, 5*fs)
    br_epoch = br_epoch.squeeze()
    #print(br_epoch.shape[2], rfft_spect_h['power_spectrogram'].shape[0])
    #exit()
    
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
        a1=np.sort(a1)
        if len(a1)>=3:
            first, second, third=a1[-1], a1[-2], a1[-3]
        elif len(a1)==2:
            first,second,third=a1[-1], a1[-2], np.min(psd)
        elif len(a1)==1:
            first,second,third=a1[-1], np.min(psd), np.min(psd)
        elif len(a1)==0:
            first,second,third=np.max(psd), np.min(psd), np.min(psd)
        
     
        #print(first, second, third)
        feat_5=first/second
        #print(feat_5)
        feat_6=first/third
        #print(feat_6)
        feat_7 = kurtosis(psd)
        #print(feat_7)
        feat_8 = skew(psd) 
        #print(feat_8)
        
        
        
        
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
X0 = np.zeros((n,1))
print(X0.shape)
feat = np.hstack((feat,X0))
print(feat.shape)

df=pd.DataFrame(feat,columns=output_cols)
df.to_csv('/media/shruti/Data/Breathing_project/breathing_notalkCRD_new.csv',index=None)
			
        
        
        

        
     
        
        
        
        
       
        

