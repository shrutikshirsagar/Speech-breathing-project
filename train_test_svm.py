#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:11:46 2020

@author: shruti
"""

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
classifier = svm.SVC(kernel='linear', probability=True,
                      random_state=random_state)
# classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                      intercept_scaling=1, l1_ratio=None, max_iter=1000,
#                      multi_class='auto', n_jobs=None, penalty='l2',
#                      random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
#                     warm_start=False)
classifier.fit(X, Y)  
x_test = pd.read_csv('/media/shruti/Data/Breathing_project/Tiles/biovad_data_20s/br_fts/0a85fd46-fada-434c-9f7a-08b81f9ed8e7.csv', header=0)

x_test = np.array(x_test)
print(x_test)
X_test = x_test[:, :-1]
print(X_test.shape)


Y_test=classifier.predict(X_test)
print(Y_test)
a = np.sum(Y_test)
print(a)
print('talking', a/X_test.shape[0]*100)
print('no_talking', (X_test.shape[0]-a)/X_test.shape[0]*100)