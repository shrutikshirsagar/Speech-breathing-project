#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:56:44 2020

@author: shruti
"""

from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from scipy import stats
x = pd.read_csv('/media/shruti/Data/Breathing_project/CRD_interspeech_new.csv', header=0)
x = np.array(x)

print(x.shape)
for i in range(15):

    X = x[:, i]
    plt.figure()
    plt.plot(X)
    plt.savefig('/media/shruti/Data/Breathing_project/CRD+interspeech/feature/'+ 'CI_' +str(i) + '.png')
    plt.clf()



