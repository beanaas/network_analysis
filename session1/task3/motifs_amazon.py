#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 1

Erik G. Larsson 2018-2020
"""

import sys
import snap
import datetime

TPATH = "/courses/TSKS33/ht2023/data/"
start_snap = datetime.datetime.now()
G = snap.LoadEdgeList(snap.PUNGraph, TPATH + "amazon0302.txt", 0, 1) #loads an undirected graph from a given edge list

t = snap.GetTriadsAll(G,-1) #get all subgraphs with three nodes, return list with the number of closed triads and the number of open triads
end_snap = datetime.datetime.now()
print ("Number of triangles", t[0]) #prints number of closed triangles
print ('Number of "open triangles"',t[2]) #prints number of open triangles
print ("Number of wedges",t[0]*3+t[2]) #prints number of wedges, all closed triangles consists of three wedges
print("time snap: ", end_snap-start_snap)

import numpy as np
import scipy.sparse as sp
start_self = datetime.datetime.now()
E = np.genfromtxt(TPATH + 'amazon0302.txt', dtype=int) #reads data from given textfile into an ndarray, read as integers
if np.min(E) == 1: #checks for 1-based indexing
    E -= 1 # python is 0 indexed
N = np.max(E)+1 #adds one because we use 0-based indexing

# undirected
#"""
r = np.concatenate((E[:,1], E[:,0])) #create array with all elements, the elements from column 1 first
c = np.concatenate((E[:,0], E[:,1])) #create array with all elements, the elements from column 0 first
#"""
# directed
"""
r = E[:,1]
c = E[:,0]
"""

r,c = zip(*set(zip(r,c))) # remove duplicates, set(zip(r,c)) creates pairs of the nodes without duplicates, * unpacks values from the set and last zip creates two lists
d = np.ones(len(r)) #creates an array of ones with the size of the length of r

A = sp.csr_array((d,(r,c)),shape=(N,N)) # creates an sparse adjacency matrix, stores the values of d at row indices r and columen indices c. The shape of the matrix is NxN

A3 = A@A@A
traceA3 = A3.trace()
number_of_triangles = (1/6)*traceA3


#wedges
u = np.ones(N)
ut = np.transpose(u)
A2 = A @ A


number_of_wedges = 0.5 *(ut @ (A2-A) @ u)

end_self = datetime.datetime.now()
print(number_of_triangles)
print(number_of_wedges)
print("time self: ", end_self-start_self)
#datetime.datetime.now()

