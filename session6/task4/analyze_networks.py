#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 6 task 4
Degree correlations

Erik G. Larsson, 2020
"""

import snap
import sys
sys.path.append("/courses/TSKS33/ht2023/common-functions")
from degreecorr import degree_corr_coeff
from degreecorr import plot_dcf
from degreecorr import plot_degreepairs

# Generate G(N,M) random graph
G = snap.GenRndGnm(snap.PUNGraph, 500, 100000, False)

# Load science collaboration network
#G = snap.LoadEdgeList(snap.PUNGraph, "collaboration.edgelist.txt", 0, 1)
#G = snap.LoadEdgeList(snap.PUNGraph, "/courses/TSKS33/ht2023/data/collaboration.edgelist.txt", 0, 1)

# Analyze nominal network
print ('degree correlation coefficient: ',degree_corr_coeff(G))
plot_degreepairs(G) 
plot_dcf(G)

# Analyze re-wired network (keeps the degree sequence)
Gr = snap.GenRewire(G, 100)
print ('degree correlation coefficient: ',degree_corr_coeff(Gr))
plot_dcf(Gr)

