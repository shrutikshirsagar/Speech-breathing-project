
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal
import scipy.io as sio
breathing=np.empty((2310,0))
num = ['11', '13', '14', '15', '16', '17', '18', '19', '26', '27', '28', '32', '34', '36', '37','38', '39', '41', '42', '44', '45', '48']

for i in num:
    br = sio.loadmat('/media/shruti/Data/crd_biovad_data/br_fin_biovad/br_sub_'+ i+ '.mat')
    print(br.keys())
    s1 = br['br_fin']
    s1 = s1.squeeze()
    s1 = s1[:, None]
    print(s1.shape)
    #a = np.delete(s1, (2311:), axis = 0)
   
    breathing=np.hstack((breathing, s1))
df=pd.DataFrame(breathing)
df.to_csv('/media/shruti/Data/breathingwithouttalking.csv',index=None, header = None)
    
