#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 hands-on session 6 task 5

Graphical Lasso applied to TCGA data

Erik G. Larsson, 2020
"""

import sys
sys.path.append("/courses/TSKS33/ht2023/common-functions")

import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLasso
from save_Gephi_gexf import save_nparray_Gephi_gexf_colors
import matplotlib.pyplot as plt

TPATH = '/courses/TSKS33/ht2023/data/'

# read TCGA data
data=pd.read_csv(TPATH+'TCGA-PANCAN-HiSeq-801x20531/data.csv')
data.drop(data.columns[0],axis=1,inplace=True)
X=data.to_numpy()

# select a subset
N1=0
N2=800
#Y=np.delete(X,range(50,500),0)
#selets subset of the data, from row N1 to row N2
Y=X[N1:N2,:]
plt.spy(Y)
plt.show()
N=Y.shape[0]
print(N)

#Computes the full covariance matrix
cov_full=np.dot(Y,np.transpose(Y))
print(np.linalg.cond(cov_full))

# run graphical Lasso
cov = GraphicalLasso(15).fit(np.transpose(Y))
plt.spy(cov.precision_)
plt.show()

# read labels to get ground truth
labels=pd.read_csv(TPATH+'TCGA-PANCAN-HiSeq-801x20531/labels.csv')
labels_array=labels['Class'].to_numpy()
labels_array=labels_array[N1:N2]
#labels_array=np.delete(labels_array_a,range(50,500),0)
labels_int=dict(zip(np.unique(labels_array),range(N)))
print(labels_int)

v=np.zeros((N,1))
for n in range(N):
    v[n]=labels_int[labels_array[n]]
    
save_nparray_Gephi_gexf_colors(cov.precision_,'TCGA.gexf',0.0000000000001,v)
