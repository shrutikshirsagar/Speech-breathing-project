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
import matplotlib.pyplot as plt

def get_br_spectrogram(br,win_):
   fs=4
   w_size =  win_ * fs
   w_shift = 1*fs   #consistant in overlap
   rfft_spect_h = ama.strfft_spectrogram(br, 4, w_size, w_shift, win_function = 'hamming' )
   
   return rfft_spect_h

def get_stft_feat_vec(br,fft_win):
	spectr_br=get_br_spectrogram(br,fft_win)
		
	# timestep X freq axis X 1
	spect_arr=spectr_br['power_spectrogram']

	#current setting with a 25 second signal from t-5 to t+20 gives 16 psd's with 3second overlaps
	ft_arr=np.empty((0,10))
	for ix in range(0,spect_arr.shape[0]):
		psd=(spectr_br['power_spectrogram'][ix,:,0])
		psd=psd/np.sum(psd)
		ft_psd=get_features(spectr_br['freq_axis'],psd)
		ft_arr=np.vstack((ft_arr,ft_psd))

	ft_vec=np.std(ft_arr,axis=0)
	
	return ft_vec

def FeatureSpectralFlatness(X, f_s):

    norm = X.mean(axis=0, keepdims=True)
    norm[norm == 0] = 1

    X = np.log(X + 1e-20)

    vtf = np.exp(X.mean(axis=0, keepdims=True)) / norm

    return (vtf)


def get_features(a_,psd_,fs_=4):

    
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

    feat_5 = kurtosis(psd_)
    #print(feat_7)
    feat_6 = skew(psd_) 

    feat_7 = FeatureSpectralFlatness(psd_, fs_)
    #print(feat_10)
    
    #in band vs out of band ratio for feat 1
    feat_8=feat_1/(feat_2+feat_3)
    
    #in band vs out of band ratio for feat 1
    feat_9=feat_2/(feat_1+feat_3)
    
    feat_10=np.sum(a_*psd_)
   
    feat_c=np.hstack((feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10))
    
    return feat_c

ix,iy=160,185
feat_csv_path='./br_fts/'
raw_vad_dir='./br_vad_raw/'
par_ids=os.listdir(feat_csv_path)
par_ids=par_ids[ix:iy]

cols=['filename','bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','kurt','sknew','flatness','ratio_25', 'ratio_50','centroid']
fft_win=10
stft_win=30
istrt=10
iend=istrt+stft_win
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

  feats_sub=np.empty((df_ft.shape[0],10))

  for ix_,row in df_ft.iterrows():
    
    fname=row['filename']
    br_sig=np.array(df_raw[fname])[4*istrt:4*iend]  #20 second signal
    br_sig=(br_sig-np.mean(br_sig))/np.std(br_sig)
    feat_vec=get_stft_feat_vec(br_sig,fft_win)
    feats_sub[ix_,:]=feat_vec
    
  df_fin=pd.DataFrame(feats_sub,columns=cols[1:])
  df_fin['filename']=np.array(df_ft['filename'])
  out_br_ft='./br_stft20_fft5_fts/'
  if not os.path.exists(out_br_ft):
     os.makedirs(out_br_ft)
  
  df_fin.to_csv(out_br_ft+Id,index=None)

  
 
   

      
      




