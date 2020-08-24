#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:53:28 2020

@author: shruti
"""

from am_analysis import am_analysis as ama
import numpy as np
import matplotlib.pyplot as plt 
#put breathing data here as a numpy array
import pandas as pd
from scipy import signal


import pandas as pd
import numpy as np
np_load_old = np.load
from sklearn import preprocessing
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # set pickle configuration to True
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc

from sklearn.model_selection import StratifiedKFold
import scipy.io as spio
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn import datasets, metrics, model_selection, svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

x = pd.read_csv('/media/shruti/Data/Breathing_project/CRD_interspeech_new.csv', header=0)
x = np.array(x)
print(x)
X = x[:, :-1]
print(X.shape)
Y = x[:,-1]
print(Y.shape)

n_classes = 2
n_samples, n_features = X.shape
print(n_classes)
s = np.count_nonzero(Y)
print('number of stress examples', s)


n_samples, n_features = X.shape
print(n_samples)
print(n_features)
# Add noisy features
random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#print(X.shape)
# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
print('stratified', cv)
# classifier = svm.SVC(kernel='linear', probability=True,
#                       random_state=random_state)
classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                      intercept_scaling=1, l1_ratio=None, max_iter=1000,
                      multi_class='auto', n_jobs=None, penalty='l2',
                      random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)
classifier.fit(X, Y)  


import pandas as pd
#output_fin=np.empty((0,12))
output_cols=['bnd_pwr_0_25','bnd_pwr_25_50','bnd_pwr_50_150','entropy','p1_p2','p1_p3','kurt','sknew','flatness','peak_f','ratio_25', 'ratio_50','ratio_150','perm_ent', 'filename', 'labels']
#load master list data
info_csv='/home/shruti/Downloads/Master_list.csv'
df_info=pd.read_csv(info_csv)[['participant_id','Shift Info','sex']]

#iterate over dataframes
for idx,row in df_info.iterrows():
	#get a subject , its shift and sex information
    sub_nm,shift,sex=str(row['participant_id']),str(row['Shift Info']),str(row['sex'])
    print(sub_nm,shift,sex)
    file_path='/media/shruti/Data/Breathing_project/Tiles/br_fts/'+sub_nm+'.csv'
    print('name',file_path)
   
    # df_br=pd.read_csv(file_path)
    try:
        x = pd.read_csv(file_path)
    except:
        continue
    x_new = np.array(x)
    print(x_new.shape[1])
    #x = x_new[:, 1:]
    x = x_new[:, :-1]
    print('x_shape', x.shape[1])
    


    Y_test =[]
    Y_test=classifier.predict(x)
    print(Y_test)

    
    
    
    Y = np.asarray(Y_test)
    Y = Y[:, None]
    print('labels', Y.shape)
    out_vec=np.hstack((x_new, Y))
    print(out_vec.shape)
    df=pd.DataFrame(out_vec, columns = output_cols)
    df.to_csv('/media/shruti/Data/Breathing_project/mask_log_reg/'+sub_nm+'.csv' ,index=None)
   

