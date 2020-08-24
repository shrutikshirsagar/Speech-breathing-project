#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:13:45 2020

@author: shruti
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:10:25 2020

@author: shruti
"""


from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal

df = pd.read_csv('/media/shruti/Data/Breathing_project/Tiles/br_vad_raw/4e541136-0db1-4749-b4c1-8dc03df3d4a5/audio_1a68_1520264962880.csv',  delimiter=',')
print(df.shape)

df = np.array(df)



br= df.squeeze()

print(br)
print(br.shape)
br=(br-np.mean(br))/np.std(br)
# plt.figure()

# plt.plot(br)

# plt.savefig('/media/shruti/Data/'+ 'tiles_brea.png')
# plt.clf()
fs=4  #sampling frequency
w_size =  15 * fs   # window size in seconds
w_shift = 1 * w_size  # window overlap

#compute short time fourier transform
rfft_spect_h = ama.strfft_spectrogram(br, fs, w_size, w_shift, win_function = 'hamming' )
power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
print(rfft_spect_h['power_spectrogram'].shape[0])

plt.figure()
   
# for ix in range(0,rfft_spect_h['power_spectrogram'].shape[0]):
#     psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
#     psd=psd/np.sum(psd)
#     plt.plot((rfft_spect_h['freq_axis']),psd)
    
# plt.savefig('/media/shruti/Data/'+ 'tiles.png')
# plt.clf()
#title='breathing_spectrogram.png' #ind



ama.plot_spectrogram_data(rfft_spect_h)
    
plt.savefig('/media/shruti/Data/'+ 'tiles_1.png')
plt.clf()