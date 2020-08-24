#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:23:36 2020

@author: shruti
"""
from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal
import scipy.io as sio
import pandas as pd
import os
import numpy as np
#output_fin=np.empty((0,9))
# output_cols=['participant_id','Shift Info','sex', ' talk', 'total', 'notalk', 'talk_to_total', 'notalk_to_total', 'talk_notalk' ]
#load master list data
folders = ['mask_3_019']#,'mask_3rule' 'mask_feat_1', 'mask_feat_1_2', 'mask_feat_2', 'mask_feat_3', 'mask_log_reg', 'mask_svm']
for j in folders:
    info_csv='/media/shruti/Data/Breathing_project/mask/'+ j + '/0a85fd46-fada-434c-9f7a-08b81f9ed8e7.csv'
    a = os.path.basename(info_csv)
    
    
    path1 = '/media/shruti/Data/Breathing_project/visulaize_mask/'+ j + '/notalk/' 
    if not os.path.exists(path1):
        os.makedirs(path1)
    # b = a.split('_')
    # c = b[1]
    # d = c.split('.')
    # e = d[0]
    print(a)
    df_info=pd.read_csv(info_csv)[['filename','labels']]
    df_info=df_info[df_info.labels==0]
    df_info=df_info.reset_index()
    
    #iterate over dataframes
    for idx,row in df_info.iterrows():
     	#get a subject , its shift and sex information
        print(idx)
        audio_nm,mask=str(row['filename']),np.array(row['labels'])
        print('mask', mask)
        
        file_path='/media/shruti/Data/Breathing_project/Tiles/br_vad_raw/'+a
       
        df = pd.read_csv(file_path)
        
        
        # x = np.array(x)
        br=np.array(df[audio_nm])
        print(br.shape)
      
        br=(br-np.mean(br))/np.std(br)
        fs=4  #sampling frequency
        w_size =  20 * fs   # window size in seconds
        w_shift = 1 * fs  # window overlap
        
        #compute short time fourier transform
        rfft_spect_h = ama.strfft_spectrogram(br, fs, w_size, w_shift, win_function = 'hamming' )
        power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
        
        
        if mask == 0:
            print(mask)
            fig = plt.figure()
            ax1 = fig.add_subplot(3,1,1)
            ax1.plot(br[15*fs:35*fs])
            
            ax2 = fig.add_subplot(3,1,2)
            for ix in range(10,21):
                psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
                psd=psd/np.sum(psd)
                plt.plot((rfft_spect_h['freq_axis']),psd)
            ax2.plot()
            
            # for ix in range(0,rfft_spect_h['power_spectrogram'].shape[0]):
            #     psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
            #     psd=psd/np.sum(psd)
            #     plt.plot((rfft_spect_h['freq_axis']),psd)
            # ax2.plot()
            ax3 = fig.add_subplot(3,1,3)
            ama.plot_spectrogram_data(rfft_spect_h)
            # Save the full figure...
            fig.savefig(path1+ audio_nm.split('.')[0] + '.png')
            
        
            fig.clf()
      