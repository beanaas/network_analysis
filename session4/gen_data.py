#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Data generation for hands-on session 4

Erik G. Larsson 2020
"""

#import sys
import snap
import numpy

TPATH = "/courses/TSKS33/ht2023/data/"
#TPATH = "./"

def genmod10star():   
    #G1 = snap.GenPrefAttach(1000, 10)
    #G1 = snap.GenRndPowerLaw(100000,2.2)
    G = snap.GenStar(snap.PUNGraph,10,False)
    G.AddEdge(4,5)
    #G = snap.GetMxScc(G1)
    N = G.GetNodes()
    
    # node numbering in the mat file is sequential n=1,2,... following the node iterator
    # gio.h line 369
#    snap.SaveMatlabSparseMtx(G, "mod10star.mat")

    # assign the degree as attribute
    x = snap.TIntFltH(N)
#    file = open("mod10star_attr.mat",'w') 
    for NI in G.Nodes():
        n=NI.GetId()
        x[n]=NI.GetDeg()
#        file.write(str(n+1)+"\t"+str(x[n])+"\n")
#        file.write(str(x[n])+"\n")
#    file.close()
    
    return G, x
 
def genLiveJournal():   
    G1 = snap.LoadEdgeList(snap.PUNGraph, TPATH + "soc-LiveJournal1.txt", 0, 1)
    snap.DelSelfEdges(G1)
    G = snap.GetMxScc(G1)
#    print(snap.IsConnected(G))
    N = G.GetNodes()

#    snap.SaveMatlabSparseMtx(G, "LiveJournal.mat")

    # assign synthetic data as attributes
    x = snap.TIntFltH(N)
#    file = open("LiveJournal_attr.mat",'w') 
    for NI in G.Nodes():
        n=NI.GetId()
        k=NI.GetDeg()
        x[n]=50000*(1+k/10.+0.1*numpy.sin(n))
#        file.write(str(n+1)+"\t"+str(x[n])+"\n")
#        file.write(str(x[n])+"\n")
 
#    file.close()
    return G, x

