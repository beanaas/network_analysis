#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a stochastic block model graph with two communities

Erik G. Larsson 2020
"""

import sys
import snap

sys.path.append("/courses/TSKS33/ht2023/common-functions")
from save_Gephi_gexf import saveGephi_gexf
from save_Gephi_gexf import saveGephi_gexf_twocolors


def gen_stoch_block_2comm(N1,N2,p1,p2,p12):

    G=snap.TUNGraph.New(N1+N2,int(3*p1*(N1+N2)))
    for n in range(N1):
        G.AddNode(n)
    for n in range(N1):
        for n1 in range(N1):
            if snap.TFlt.GetRnd()<p1 and n1>n:
                G.AddEdge(n,n1)
    for n in range(N2):
        G.AddNode(N1+n)
    for n in range(N2):
        for n1 in range(N2):
            if snap.TFlt.GetRnd()<p2 and n1>n:
                G.AddEdge(N1+n,N1+n1)
    for n in range(N1):
        for n1 in range(N2):
            if snap.TFlt.GetRnd()<p12:
                G.AddEdge(n,N1+n1)

#    snap.PrintInfo(G)
#    
#    iA=snap.TIntV(N1)
#    iB=snap.TIntV(N2)
#    for n in range(N1):
#        iA[n]=n
#    for n in range(N2):
#        iB[n]=N1+n
#
#    saveGephi_gexf_twocolors(G,"SB-network.gexf",iA,iB)

    return G

