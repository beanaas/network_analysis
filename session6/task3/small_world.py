#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 hands-on session 6 task 4

Small world experiments

Erik G. Larsson, 2018-2020
"""

import snap

N=1000

# circle network
print("--- circle network ---")
G1=snap.GenSmallWorld(N, 2, 0)
print (snap.GetBfsEffDiam(G1, 250, False))
G1r=snap.GenRewire(G1, 100)
print (snap.GetBfsEffDiam(G1r, 250, False))

# Watt-Strogatz, 0.01
print("--- WS 0.01 ---")
G2=snap.GenSmallWorld(N, 2, 0.01)
print (snap.GetBfsEffDiam(G2, 250, False))
G2r=snap.GenRewire(G2, 100)
print (snap.GetBfsEffDiam(G2r, 250, False))

# Watt-Strogatz, 0.1
print("--- WS 0.1 ---")
G3=snap.GenSmallWorld(N, 2, 0.1)
print (snap.GetBfsEffDiam(G3, 250, False))
G3r=snap.GenRewire(G3, 100)
print (snap.GetBfsEffDiam(G3r, 250, False))

# scale-free network
print("--- scale-free ---")
G4=snap.GenRndPowerLaw (N, 2.2, True)
snap.PrintInfo(G4, "scale-free")
snap.PlotInDegDistr(G4,"scale-free","scale-free")
print (snap.GetBfsEffDiam(G4, 250, False))
G4r=snap.GenRewire(G4, 100)
print (snap.GetBfsEffDiam(G4r, 250, False))

# Amazon network
print("--- amazon ---")
G5=snap.LoadEdgeList(snap.PUNGraph, "/courses/TSKS33/ht2023/data/amazon0302.txt", 0, 1)
snap.PrintInfo(G5, "amazon")
snap.PlotInDegDistr(G5,"amazon","amazon")
print (snap.GetBfsEffDiam(G5, 250, False))
G5r=snap.GenRewire(G5, 100)
print (snap.GetBfsEffDiam(G5r, 250, False))

