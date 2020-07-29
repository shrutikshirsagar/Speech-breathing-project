#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:55:30 2020

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
outputfile = '/media/shruti/Data/breathing_spectrogram/ab'
x = pd.read_csv('/media/shruti/Data/breathing_spectrogram/breathing_no_talk.csv', header=0)
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
#                      random_state=random_state)
# classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                     intercept_scaling=1, l1_ratio=None, max_iter=1000,
#                     multi_class='auto', n_jobs=None, penalty='l2',
#                     random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
#                     warm_start=False)

classifier = RandomForestClassifier(max_depth=2, random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
print('mean',mean_fpr)
fig, ax = plt.subplots()
acc1 = []
F1 = []
precision =[]
recall = []
for i, (train, test) in enumerate(cv.split(X, Y)):
    classifier.fit(X[train], Y[train])
    y_pred=classifier.predict(X[test])
    print('x_train number', X[train].shape)
    print('Y_train number', Y[train].shape)
    viz =  metrics.plot_roc_curve(classifier, X[test], Y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    print('x_test number', X[test].shape)
    print('Y_test number', Y[test].shape)
    sp = np.count_nonzero(Y[test])
    print('number of breathing examples', sp)
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    y_score = classifier.fit(X[train], Y[train]).decision_function(X[test])
    a =roc_auc_score(Y[test], y_score)
    print(a)
    acc=100*sum((y_pred-Y[test])==0)/len(Y[test])
    print('accuracy', acc) 
    acc1.append(acc)
    print(classification_report(Y[test],y_pred))
    print('binary', precision_recall_fscore_support(Y[test],y_pred, average='binary'))
    print('macro', precision_recall_fscore_support(Y[test],y_pred, average='macro'))
    print('micro', precision_recall_fscore_support(Y[test],y_pred, average='micro'))
    print('weighted', precision_recall_fscore_support(Y[test],y_pred, average='weighted'))
    F12 = precision_recall_fscore_support(Y[test],y_pred, average='macro')
    F1.append(F12[2])
    recall.append(F12[1])
    precision.append(F12[0])
    class_names = ['stress','non-stress']
    cm = confusion_matrix(Y[test],y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()

print(F1)
b = statistics.mean(F1)
print('FINAL f1 SCORE', b)
print(acc1)
new = statistics.mean(acc1)
print('Final accuracy', new)
c = statistics.mean(recall)
print('recall mean', c)
cd = statistics.mean(precision)
print('precision mean', cd)
f = open(outputfile, "a")
f.write("--------------------------------------------------------------\n")
f.write("SVM results\n")
print("results of   %f (stress sample), %f (acc),%f (F1 score), %f (precision), %f (recall), %f (AUC)" % (s,b,new,c,cd,mean_auc))
print()

f.close()
