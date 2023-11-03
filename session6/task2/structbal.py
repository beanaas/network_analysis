#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Look at some triangles in a signed network and see if structural balance seems to hold 

Erik G. Larsson, 2018-2020
"""

import sys
import snap
from random import randint

def getsign(n1,n2):    
    p=0
    n=0
    if Gp.IsEdge(n1,n2):
        p=1
    if Gn.IsEdge(n1,n2):
        n=-1
    return p+n
        
G = snap.TUNGraph.New()  # all edges
Gp = snap.TUNGraph.New() # positive edges
Gn = snap.TUNGraph.New() # negative edges

#f = open('soc-sign-epinions.txt')
f = open('/courses/TSKS33/ht2023/data/soc-sign-epinions.txt')
i=1
Nneglected=0
for line in f:
        s = line
        if not s[0]=='#':
            x = s.split("\t")
            n1 = int(x[0])
            if not G.IsNode(n1):
                G.AddNode(n1)
                Gp.AddNode(n1)
                Gn.AddNode(n1)
            n2 = int(x[1])
            if not G.IsNode(n2):
                 G.AddNode(n2)
                 Gp.AddNode(n2)
                 Gn.AddNode(n2)

            if (n1==n2):
                continue
            
            sign = int(x[2])

            if G.IsEdge(n1,n2):
                if Gp.IsEdge(n1,n2) and (sign==-1):
                    # print n1, n2, "already exists with different sign"
                    Nneglected = Nneglected+1
                    Gp.DelEdge(n1,n2)

                if Gn.IsEdge(n1,n2) and (sign==1):
                    # print n1, n2, "already exists with different sign"
                    Nneglected = Nneglected+1
                    Gn.DelEdge(n1,n2)                             
            else:
                G.AddEdge(n1,n2)
                if (sign==1):
                    Gp.AddEdge(n1,n2)
                else:
                    Gn.AddEdge(n1,n2)

                if (i%50000==0):
                    print (i)
            i=i+1


snap.PrintInfo(G, "signed network", "/dev/stdout", False)
snap.PrintInfo(Gp, "signed network", "/dev/stdout", False)
snap.PrintInfo(Gn, "signed network", "/dev/stdout", False)

print ("total number of edges:", i-1)
print ("number of inconsistent edges:", Nneglected)


# sample some random triangles
for i in range(1,500):
    n1 = G.GetRndNId() # random node
    n1i = G.GetNI(n1)
    if n1i.GetDeg()>=2:
        d = n1i.GetDeg()
        i1 = randint(0,d-1)
        while 1==1:
            i2 = randint(0,d-1)
            if not i1==i2:
                break
        n2 = n1i.GetNbrNId(i1)
        n3 = n1i.GetNbrNId(i2)
        if not G.IsEdge(n2,n3):
            continue
        
        s12 = getsign(n1,n2)
        s13 = getsign(n1,n3)        
        s23 = getsign(n2,n3)
        
        print ("=========================")
        print ("edge ", n1, " - ", n2, " has sign ", s12)
        print ("edge ", n1, " - ", n3, " has sign ", s13)
        print ("edge ", n2, " - ", n3, " has sign ", s23)
 
        if (s12*s13*s23==0):
            print ("The triangle", n1, n2, n3, " contains at least one inconsistent edge, ignoring it.")
            continue
        
        if (s12*s13*s23==1):
            print ("The triangle", n1, n2, n3, "is strongly balanced.")
        else:
            print ("The triangle", n1, n2, n3, "is unbalanced.")
    
