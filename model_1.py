#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:41:45 2020

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

x = pd.read_csv('/media/shruti/Data/Breathing_project/Om.csv', header=0)
x = np.array(x)
print(x.shape)
X = x[:, :-1]
print(X.shape)
Y = x[:,-1]
print(Y)

n_classes = 2

s = np.count_nonzero(Y)
print('number of stress examples', s)



random_state = np.random.RandomState(0)

cv = StratifiedKFold(n_splits=5)
print('stratified', cv)

classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                      intercept_scaling=1, l1_ratio=None, max_iter=1000,
                      multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)

# classifier = svm.SVC(kernel='linear', probability=True,
#                       random_state=random_state)

x_test = pd.read_csv('/media/shruti/Data/Breathing_project/CRD_interspeech.csv', header=0)
x_test1 = np.array(x_test)
print(x_test1.shape)
x_test2 = x_test1[:, :-1]
print(x_test2.shape)
# x_test = x_test[:, 1:]
# print("total smaples", x_test.shape)
# classifier.fit(X, Y)
# y_pred=classifier.predict(x[:,:-1])
# a = np.sum(y_pred)
# print(a)
classifier.fit(X, Y)
y_pred=classifier.predict(x_test2)
a = np.sum(y_pred)
print(a)
Y_test = x_test1[:,-1]

acc=100*sum((y_pred-Y_test)==0)/len(Y_test)
print('accuracy', acc) 