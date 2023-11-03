import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from matplotlib import pyplot as plt

filename = "verification"
nr_users = 2000
nr_movies = 1500

def load_data(name):
    data = np.genfromtxt(name,delimiter=',',dtype=int)
    data[:,0:2] -= 1
    return data

def getA(data):
    nr_ratings = len(data)

    r = np.concatenate((np.arange(nr_ratings,dtype=int), np.arange(nr_ratings,dtype=int)))
    c = np.concatenate((data[:,0], data[:,1]+nr_users))
    d = np.ones((2*nr_ratings,))

    A = sp.csr_matrix((d,(r,c)),shape=(nr_ratings,nr_users+nr_movies))

    return A

training_data = load_data(filename+'.training')
test_data = load_data(filename+'.test')
