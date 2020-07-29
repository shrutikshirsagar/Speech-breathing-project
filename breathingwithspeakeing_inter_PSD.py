"""Code to compute and generate the spectrogram for breathing using the amplitude modulation toolbox"""
from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal
label = 'upper_belt'
df = pd.read_csv('/media/shruti/Data/Breathing/lab/labels.csv', delimiter=',')
print(df.shape)
#y_devel = df[label][df['filename'].str.startswith('devel')].values
#print('deve', y_devel.shape)
num = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14','15', '16']
breathing=np.empty((0,6000))
for i in num:
    print(i)
    df2 = df.loc[df['filename'] == 'train_'+ i+ '.wav']
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
        
    plt.savefig('/media/shruti/Data/Breathing/PSD/inter/'+'train_'+ i+ '.png')
    plt.clf()
#title='breathing_spectrogram.png' #ind
#plt.savefig(title)
