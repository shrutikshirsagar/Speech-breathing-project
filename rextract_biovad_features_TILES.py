#!/usr/bin/env python

""" 
Reextrated some additional features for OMSignal data
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pyentrp import entropy as ent
from am_analysis import am_analysis as ama
from scipy import interpolate, signal
import os
import pytz
from datetime import datetime
from scipy import signal
import scipy.io as sio
from scipy.signal import periodogram, welch
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
import scipy.fftpack as fftpack
from pyentrp import entropy as ent

def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.log(X + 1e-20)

    vtf = np.exp(X.mean(axis=0, keepdims=True)) / norm

    return (vtf)

def get_psd_ama(x_,fs):
	#computing signal fft
	N = len(x_)
	# sample spacing
	w_size =  20 * fs
	w_shift=  20*fs
	
	rfft_spect_h = ama.strfft_spectrogram(x_, 4, w_size, w_shift, win_function = 'hamming' )
	power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
	
	psd=(rfft_spect_h['power_spectrogram'][0,:,0])
	psd=psd/np.sum(psd)
	freq=rfft_spect_h['freq_axis']
	
	
	return freq, psd

def get_features(x_,fs_):

    a_,psd_=get_psd_ama(x_,fs_)
    psd_=psd_/np.sum(psd_)
    b = np.where(a_<0.25) 
    feat_1 = psd_[b]
    feat_1 = np.sum(feat_1)
  
    c = np.where((a_ >0.25) & (a_<0.5))
    feat_2 = psd_[c]
    feat_2 = np.sum(feat_2)

    d = np.where((a_ >0.5) & (a_<1.5))
    feat_3 = psd_[d]
    feat_3 = np.sum(feat_3)

    feat_4 = -np.sum(psd_*np.log(psd_))

    peaks, properties = find_peaks(psd_)
    a1 = psd_[peaks]
    a1=np.sort(a1)
    if len(a1)>=3:
        first, second, third=a1[-1], a1[-2], a1[-3]
    elif len(a1)==2:
	    first,second,third=a1[-1], a1[-2], np.min(psd_)
    elif len(a1)==1:
    	first,second,third=a1[-1], np.min(psd_), np.min(psd_)
    elif len(a1)==0:
    	first,second,third=np.max(psd_), np.min(psd_), np.min(psd_)
 
    #print(first, second, third)
    feat_5=first/second
    #print(feat_5)
    feat_6=first/third
    #print(feat_6)
    feat_7 = kurtosis(psd_)
    #print(feat_7)
    feat_8 = skew(psd_) 

    feat_10 = FeatureSpectralFlatness(psd_, fs_)
    #print(feat_10)
    
    feat_11 = a_[np.argmax(psd_)]
    
    #in band vs out of band ratio for feat 1
    feat_12=feat_1/(feat_2+feat_3)
    
    #in band vs out of band ratio for feat 1
    feat_13=feat_2/(feat_1+feat_3)
    
    #in band vs out of band ratio for feat 1
    feat_14=feat_3/(feat_1+feat_2)
    
    #get timeseries entropy (PE)
    lag=1
    feat_15=ent.permutation_entropy(x_,3,lag)
    
   
    
    feat_c=np.hstack((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_10, feat_11,feat_12,feat_13,feat_14,feat_15))
    
    return feat_c

ix,iy=0,212
feat_csv_path='./br_fts/'
raw_vad_dir='./br_vad_raw/'
par_ids=os.listdir(feat_csv_path)
par_ids=par_ids[ix:iy]

cols=['filename','bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','p1_p2','p1_p3','kurt','sknew','flatness','peak_f','ratio_25', 'ratio_50','ratio_150','perm_ent']

for i, Id in enumerate(par_ids):
  #skipping out subject which have their files
 
  print("Subject number being processed is ",i+ix)  
  print('Id: ',Id) 

  
  f_name_ft=feat_csv_path+'/'+Id
  f_name_raw=raw_vad_dir+'/'+Id
  

  df_ft=pd.read_csv(f_name_ft)
  try:
  
      df_raw=pd.read_csv(f_name_raw) 
  except:
      continue

  feats_sub=np.empty((df_ft.shape[0],14))

  for ix_,row in df_ft.iterrows():
    
    fname=row['filename']
    br_sig=np.array(df_raw[fname])[4*15:4*35]  #20 second signal
    feat_vec=get_features(br_sig,4)
    feats_sub[ix_,:]=feat_vec
    
  df_fin=pd.DataFrame(feats_sub,columns=cols[1:])
  df_fin['filename']=np.array(df_ft['filename'])
  out_br_ft='./biovad_data_20s/'+'br_fts/'
  if not os.path.exists(out_br_ft):
  	os.makedirs(out_br_ft)
 
  df_fin.to_csv(out_br_ft+Id,index=None)

  
 
   

      
      




