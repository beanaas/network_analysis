import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from matplotlib import pyplot as plt

filename = "frena297"
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

def getR(data):
    u = data[:,0]
    m = data[:,1]
    r = data[:,2]
    R = sp.csr_matrix((r,(u,m)),shape=(nr_users,nr_movies))

    return R

def getRHAT(pairs, r_average, bu, bm):
    rhat = np.zeros((nr_users, nr_movies))

    for user, movie in pairs:
        val = (r_average+bu[user]+bm[movie]).round(3)
        if(val>5):
            val = 5
        elif(val<1):
            val = 1
        rhat[user][movie] = val
    return rhat

def train(training_data, r_average):
    training_data = load_data(filename+'.training')
    test_data = load_data(filename+'.test')
    u = training_data[:,0]
    m = training_data[:,1]
    r = training_data[:,2]
    r_average = r.sum()/r.size
    rmatrix = getR(training_data).toarray()
    A = getA(training_data)
    c = r-r_average
    At = A.transpose()
    b = np.linalg.lstsq((At@A).toarray(), At@c)[0]
    bu = b[:nr_users]
    bm = b[nr_users:]
    return bu, bm

def getRMSE(pairs,rmatrix, rhat):
    C = len(pairs)
    tmp = []
    abs_errors = []
    for user, movie in pairs:
        tmp.append(((rmatrix[user][movie]-rhat[user][movie])**2)/C)
        abs_error = (rmatrix[user][movie]-rhat[user][movie].round(0))

        abs_errors.append(abs_error)
    
    RMSE = sum(tmp)**(1/2)
    return RMSE, abs_errors

def predictTest(bu, bm):
    test_data = load_data(filename+'.test')
    u = test_data[:,0]
    m = test_data[:,1]
    r = test_data[:,2]
    r_average = r.sum()/r.size
    rmatrix = getR(test_data).toarray()
    pairs = list(zip(u,m))
    rhat = getRHAT(pairs, r_average, bu, bm)
    RMSE, abs_errors = getRMSE(pairs, rmatrix, rhat)
    print("RMSE: ", RMSE)
    plt.title('ABS errors')
    plt.hist(abs_errors, bins = [1,2,3,4,5])
    plt.show()



if __name__ == '__main__':
    
    bu, bm = train()


    predictTest(bu, bm)







    

test_data = load_data(filename+'.test')


