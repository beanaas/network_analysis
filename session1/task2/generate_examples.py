#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 1

Erik G. Larsson 2018-2020
"""

import sys
sys.path.append("/courses/TSKS33/ht2023/common-functions")
import snap
import os
from save_Gephi_NET import saveGephi_net

TPATH = "/courses/TSKS33/ht2023/data/"

# ==========================

# Generate a complete graph with 5 nodes (K5) and look at it via Graphviz
G = snap.GenFull(snap.PUNGraph, 5)#snap.PUNGraph is a type of graph and 5 is the number of nodes in the graph. PUNGraph is undirected graph
#for EI in G.Edges():
#    print "link from %d to %d" % (EI.GetSrcNId(), EI.GetDstNId())
    
snap.SaveGViz(G, "_undirected-completely-connected.dot", "Undirected Completely Connected Network", True)#saves the graph G as a dot file
os.system("neato -Tpdf _undirected-completely-connected.dot >_undirected-completely-connected.pdf") #creates a pdf representation for the graph

# ==========================

# Generate a star graph and visualize in Graphviz
G = snap.GenStar(snap.PNGraph, 10, True) #a directed graph in the shape of star with 10 nodes, one central node with 9 leaf nodes. The True means it is connected to itself
#for EI in G.Edges():
#    print "edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
    
snap.SaveGViz(G, "_directed-star.dot", "Directed Star Graph", True) #same as before
os.system("neato -Tpdf _directed-star.dot >_directed-star.pdf") #-----------||---------

# ==========================

# Generate some Poisson random graphs and visualize with Gephi
G1 = snap.GenRndGnm(snap.PUNGraph, 100, 50) #random undirected graph with 100 nodes and 50 edges
G2 = snap.GenRndGnm(snap.PUNGraph, 100, 100) #random undirected graph with 100 nodes and 100 edges
G3 = snap.GenRndGnm(snap.PUNGraph, 100, 1000) #random undirected graph with 100 nodes and 1000 edges

saveGephi_net(G1,"_Poisson-1.NET") #saves it in a format that can be used in gephi
saveGephi_net(G2,"_Poisson-2.NET")
saveGephi_net(G3,"_Poisson-3.NET")

G4 = snap.GenRndGnm(snap.PUNGraph, 1000, 10000) #random undirected graph with 1000 nodes and 10000 edges
snap.PlotInDegDistr(G4, "_Poisson-4", "Poisson, degree distribution") #creates a plot showing the distribution of nodes with different degrees

# ==========================

# Generate a scale-free network and visualize with Gephi
G1 = snap.GenPrefAttach(10, 3) #generates an undirected graph with power-law degree distribution the desired amounts of nodes are 10 and the desired degree for each node is 3
G2 = snap.GenPrefAttach(50, 5) #--||-- desired amounts of nodes are 50 and the desired degree for each node is 5
G3 = snap.GenPrefAttach(100, 10) #--||-- desired amounts of nodes are 100 and the desired degree for each node is 10

saveGephi_net(G1,"_Pref-attach-1.NET") #saves it in a format that can be used in gephi
saveGephi_net(G2,"_Pref-attach-2.NET") #saves it in a format that can be used in gephi
saveGephi_net(G3,"_Pref-attach-3.NET") #saves it in a format that can be used in gephi
snap.PrintInfo(G3, "Python type PNGraph", "_Pref-attach-3.info.txt", False) #saves some information about graph into a textfile

G4 = snap.GenPrefAttach(100000, 10) #generates an undirected graph with power-law degree distribution the desired amounts of nodes are 100000 and the desired degree for each node is 10
snap.PlotInDegDistr(G4, "_Pref-attach-4", "Preferential attachment 4, degree distribution") #creates a plot showing the distribution of nodes with different degrees

# ==========================

# Examine the 88234-edge subnetwork of the Facebook graph
# http://snap.stanford.edu/data/ego-Facebook.html
G = snap.LoadEdgeList(snap.PUNGraph, TPATH + "facebook_combined.txt", 0, 1) #loads an undirected graph from a given edge list, 0 means start node is first in the edge list
snap.PrintInfo(G, "facebook", "_facebook-info.txt", False) #saves some information about graph into a textfile

# ==========================

# Examine the Amazon product co-purchase network
# http://snap.stanford.edu/data/amazon0302.html
G = snap.LoadEdgeList(snap.PUNGraph, TPATH + "amazon0302.txt", 0, 1) #loads an undirected graph from a given edge list
snap.PrintInfo(G, "amazon", "_amazon0302-info.txt", False) #saves some information about graph into a textfile

# ==========================

# Examine DBLP
G = snap.LoadEdgeListStr(snap.PUNGraph, TPATH + "DBLP.txt", 0, 1,Mapping=False) #another way to load an undirected graph from a given edge list
snap.PrintInfo(G, "DBLP", "_dblp-info.txt", True) #saves some information about graph into a textfile that already exists

# Examine by some random sampling if the "small-world" property of DBLP seems to be true.
for i in range(1,25):
    N1 = G.GetRndNId(); #random node id from the graph G
    N2 = G.GetRndNId(); #random node id from the graph G

    L = snap.GetShortPath(G, N1, N2) #get the shortest path length from the random ids in the Graph
    print (L)
