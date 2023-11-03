import snap 
import numpy as np
import matplotlib.pyplot as plt


for filenr in range(1,9):
    file = str(filenr)

    G = snap.LoadEdgeList(snap.PUNGraph,'/courses/TSKS33/ht2023/data/session6-task1/g'+file+'.edges')

    clust = G.GetNodeClustCfAll()

    f = open('G-clustering-'+file+'.csv', 'w')
    print('NodeId,Degree,ClustCoeff',file=f)
    for n in G.Nodes():
        nID = n.GetId()    
        print('{},{},{:2.15f}'.format(nID,n.GetDeg(),clust[nID]),file=f)
    f.close()



    degs = G.GetDegCnt()
    N = G.GetNodes()
    c = 0
    f = open('G-degree-'+file+'.csv','w')
    for i in range(len(degs)):
        d = degs[i].Val1.Val
        n = degs[i].Val2.Val
        c = c + n
        print('{},{},{}'.format(d,n/N,c/N),file=f)
    f.close()

    C = np.genfromtxt('G-clustering-'+file+'.csv',delimiter=',')
    plt.figure()
    plt.scatter(C[:,1],C[:,2])
    plt.xlabel('degree')
    plt.ylabel('clustering coefficient')
    plt.title('File: '+file)
    plt.savefig('Deg-Clust-'+file+'.png')

    D=np.genfromtxt('G-degree-'+file+'.csv', delimiter=',')
    if len(D.shape) == 1:
        D = D[None]
    plt.figure()
    plt.plot(D[:,0],D[:,1],label='degree distribution')
    plt.plot(D[:,0],D[:,2],label='cumulative degreedistribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree')
    plt.legend()
    plt.title('File: '+file)
    plt.savefig('Deg-Dist-'+file+'.png')
        
