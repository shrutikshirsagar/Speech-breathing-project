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
breathing=np.empty((6000,0))
#y_devel = df[label][df['filename'].str.startswith('devel')].values
#print('deve', y_devel.shape)
num = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14','15']

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
    
    breathing=np.hstack((breathing, s1))
df=pd.DataFrame(breathing)
df.to_csv('/media/shruti/Data/breathing_devel.csv',index=None, header = None)
