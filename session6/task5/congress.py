#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 hands-on session 6 task 5

Graphical Lasso applied to U.S. Congress voting data

Erik G. Larsson, 2020
"""

import sys
sys.path.append("/courses/TSKS33/ht2023/common-functions")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import graphical_lasso
from save_Gephi_gexf import save_nparray_Gephi_gexf_colors
#import scipy.io as sio

# Setup
VOTE = {"Nay": 1, "Yea": 2, "Not Voting": -1, "No": 1, "Aye": 2, "Present": -2}
PARTY = {"Republican": 1, "Democrat": 0}

PID = {} # senator ID -> index starting at zero
NAMES = {} # senator name
partys={} # senate party membership

# Read and parse the voting data
N = 0
X = np.zeros([0,0],dtype=int)  # array with votes
for i in range(0,699):
    data=pd.read_pickle('/courses/TSKS33/ht2023/data/US-congress-votes/vote-116-2019-H'+str(i)) 
    data.drop(columns=['state','district'],inplace=True)
    votes = data.to_numpy()
        
    X=np.append(X,np.zeros([N,1],dtype=int),axis=1)

    # parse the database
    for j in range(votes.shape[0]):
        pid = votes[j,0]
        if pid in PID:
            n=PID[pid]
        else:
            n=N
            PID[pid]=n
            NAMES[n]=votes[j,2]
            partys[n]=PARTY[votes[j,3]]
            N=N+1
            X=np.append(X,np.zeros([1,i+1],dtype=int),axis=0)
          #  print(vote_matrix)
    #    print(pid,n)

        X[n,i] = VOTE[votes[j,1]]
     #   print(vote_matrix)

print(N)
        
# Extract subset of senators, and pre-process data.
# Treat missing votes as "no"
Y = np.zeros_like(X,dtype=float)
Y[X==1]=0 # no -> 0
Y[X==2]=1 # yes -> 1
N=200
Y=Y[0:N,:]  

# Visually inspect the raw data
plt.spy(Y)
plt.show()

# Use the Banerjee heuristic to form the sample covariance
Z=np.cov(Y,bias=1)  + 1/3.0 * np.eye(N)
#Z=np.dot(Y,np.transpose(Y))/N  + 1/3.0 * np.eye(N)

# Run graphical Lasso on sample covariance matrix Z
covariance, precision =graphical_lasso(Z,alpha=.12,max_iter=100)
#print(np.around(precision, decimals=3)) 

# Color the nodes according to ground truth (republican/democrat)
v=np.zeros([N,1],dtype=int)
for i in range(N):
    v[i]=partys[i]

save_nparray_Gephi_gexf_colors(precision,'congress.gexf',0.0000000000001,v)
#print(np.nonzero(precision)[0].shape)

plt.spy(precision)
plt.show()
