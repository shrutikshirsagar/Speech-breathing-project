"""Code to compute and generate the spectrogram for breathing using the amplitude modulation toolbox"""
from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal
import scipy.io as sio
num = ['11', '13', '14', '15', '16', '17', '18', '19', '26', '27', '28', '32', '34', '36', '37','38', '39', '41', '42', '44', '45', '48']

for i in num:
    br = sio.loadmat('/media/shruti/Data/crd_biovad_data/br_fin_biovad/br_sub_'+ i+ '.mat')
    print(br.keys())
    s1 = br['br_fin']
    s1 = s1.squeeze()

    print(s1.shape)
    br= s1
    br=(br-np.mean(br))/np.std(br)
    fs=4  #sampling frequency
    w_size =  15 * fs   # window size in seconds
    w_shift = 1 * w_size  # window overlap

    #compute short time fourier transform
    rfft_spect_h = ama.strfft_spectrogram(br, fs, w_size, w_shift, win_function = 'hamming' )
    power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
    print(rfft_spect_h['power_spectrogram'].shape[0])
    
    plt.figure()
   
    for ix in range(0,rfft_spect_h['power_spectrogram'].shape[0]):
        psd=(rfft_spect_h['power_spectrogram'][ix,:,0])
        psd=psd/np.sum(psd)
        plt.plot((rfft_spect_h['freq_axis']),psd)
        
    plt.savefig('/media/shruti/Data/Breathing/PSD/CRD/'+'br_sub_'+ str(i)+ '.png')
    plt.clf()


