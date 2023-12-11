#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to compute the degree correlation coefficient and function

Erik G. Larsson, 2020
"""

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# calculate degree correlation
def degree_corr_coeff(G):    
    Ek = 0.0
    M = 0
    #iterate over all edges in graph
    #gets src and dst node, get degrees of both
    for EI in G.Edges():
        n1 = EI.GetSrcNId()        
        n2 = EI.GetDstNId()
#     print "edge (%d, %d)" % (n1,n2)
        k1 = G.GetNI(n1).GetDeg()
        k2 = G.GetNI(n2).GetDeg()
#     print "degrees (%d, %d)" % (d1,d2)
        M = M+1
        Ek = Ek + k1 + k2
    #divide the degree by 2*M because counted twice
    Ek = Ek/(2*M)
     
    Ck = 0.0
    Vk = 0.0
    #iterates over edges, gets the numerator and the denominatpr
    for EI in G.Edges():
        n1 = EI.GetSrcNId()        
        n2 = EI.GetDstNId()
        k1 = G.GetNI(n1).GetDeg()
        k2 = G.GetNI(n2).GetDeg()
#        print "degrees (%d, %d)" % (k1,k2)
        Ck = Ck + (k1-Ek)*(k2-Ek)
        Ck = Ck + (k2-Ek)*(k1-Ek)
        Vk = Vk + (k1-Ek)*(k1-Ek)
        Vk = Vk + (k2-Ek)*(k2-Ek)
            
    print ("Ek=",Ek)
    print ("Ck=",Ck)
    print ("Vk=",Vk)
    #calculates the coefficient
    r = Ck/Vk
    return r 

# plot the degree correlation function
def plot_dcf(G):
    kmax=0
    #iterate over nodes, finds max nodedegree
    for NI in G.Nodes():
        if NI.GetDeg()>kmax:
            kmax=NI.GetDeg()
    #array of zeros size of maxdegree
    sum_deg=np.zeros((kmax+1))
    Nk=np.zeros((kmax+1,1))
    #iterates over nodes
    for NI in G.Nodes():
        ad=0;
        d=NI.GetDeg()
        if d>0:
            #iterates over the neighbors
            for i in range(0,d):
                #add normalized degree
                ad = ad + float(G.GetNI(NI.GetNbrNId(i)).GetDeg()) / float(d);
            #update values
            sum_deg[d]=sum_deg[d]+ad;
            Nk[d]=Nk[d]+1
    #final update of array
    dcf=np.zeros((kmax+1))
    for k in range(kmax+1):
        if Nk[k]>0:
            dcf[k]=sum_deg[k]/Nk[k]
    #plot it
    krange=range(kmax+1)
    plt.scatter(krange,dcf)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1,1000)
    plt.ylim(1,1000)
   # ax.set_xlim([0,100])
    plt.show()
    
    return krange, dcf

# generate scatter plot of degree pairs for all links
def plot_degreepairs(G):
    
    M=G.GetEdges()
    x=np.zeros((2*M,1))
    y=np.zeros((2*M,1))
    i=0
    #iterates over all edges, adds them to list so they can be plotted
    for EI in G.Edges():
        n1 = EI.GetSrcNId()
        n2 = EI.GetDstNId()
        d1 = G.GetNI(n1).GetDeg()
        d2 = G.GetNI(n2).GetDeg()
        x[i], y[i]=d1,d2
        x[i+1], y[i+1]=d2,d1
        i=i+2  
    plt.scatter(x,y)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
