import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from matplotlib import pyplot as plt

filename = "albintest"
#nr_users = 2000
#nr_movies = 1500

nr_users = 10
nr_movies = 5

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

def show_histogram(abs_errors):
    plt.title('ABS errors train')
    plt.hist(abs_errors, bins=[0,1,2,3,4,5])
    plt.show()

def getRMSE(pairs,rmatrix, rhat):
    C = len(pairs)
    tmp = []
    abs_errors = []
    for user, movie in pairs:
        tmp.append(((rmatrix[user][movie]-rhat[user][movie])**2)/C)
        abs_error = int(abs((rmatrix[user][movie]-rhat[user][movie].round(0))))
 
        abs_errors.append(abs_error)
    RMSE = sum(tmp)**(1/2)
    return RMSE, abs_errors


class BaseLinePredictor:
    def __init__(self):
        self.training_data = load_data(filename+'.training')
        self.test_data = load_data(filename+'.test')
        self.bm = None
        self.bu = None
        self.train_RMSE = None
        self.train_abs_errors = None
        self.test_RMSE = None
        self.test_abs_errors = None
        self.train_rhat = None
        self.test_rhat = None
    
    def predict(self, r_average, pairs):
        rhat = np.zeros((nr_users, nr_movies))
        for user, movie in pairs:
            val = (r_average+self.bu[user]+self.bm[movie]).round(3)
            if(val>5):
                val = 5
            elif(val<1):
                val = 1
            rhat[user][movie] = val
        return rhat
    
    def test(self):
        u = self.test_data[:,0]
        m = self.test_data[:,1]
        r = self.test_data[:,2]
        r_average = r.sum()/r.size
        rmatrix = getR(self.test_data).toarray()
        pairs = list(zip(u,m))
        self.test_rhat = self.predict(r_average, pairs)
        self.test_RMSE, self.test_abs_errors = getRMSE(pairs, rmatrix, self.test_rhat)

    def train(self):
        u = self.training_data[:,0]
        m = self.training_data[:,1]
        r = self.training_data[:,2]

        pairs = list(zip(u,m))
        r_average = r.sum()/r.size
        rmatrix = getR(self.training_data).toarray()

        A = getA(self.training_data)
        c = r-r_average
        At = A.transpose()
        b = np.linalg.lstsq((At@A).toarray(), At@c)[0]
        self.bu = b[:nr_users]
        self.bm = b[nr_users:]

        self.train_rhat = self.predict(r_average, pairs)
        self.train_RMSE, self.train_abs_errors = getRMSE(pairs, rmatrix, self.train_rhat)
        

class NeighborhoodPredictor:
    def __init__(self):
        self.training_data = load_data(filename+'.training')
        #self.test_data = load_data(filename+'.test')
        self.bm = None
        self.bu = None
        self.train_RMSE = None
        self.train_abs_errors = None
        self.test_RMSE = None
        self.test_abs_errors = None
        self.train_rhat = None
        self.test_rhat = None
        self.tran_rtilde = None
        


    def predict(self, r_average, pairs):

        rmatrix = getR(self.training_data)
        ci = []
        ri =[]
        data = []
        for user, movie in pairs:
            val = (r_average+self.bu[user]+self.bm[movie]).round(3)
            if(val>5):
                val = 5
            elif(val<1):
                val = 1
            ri.append(user)
            ci.append(movie)
            data.append(val)

        rhat = sp.csr_matrix((data,(ri,ci)),shape=(nr_users,nr_movies))

        i = 0
        rtilde = (rmatrix-rhat).tocsc()
        colIndex = []
        rowIndex = []
        vals = []
        
        rowD = []
        colD = []
        dataD = []


        for movie1 in range(nr_movies):
            for movie2 in range(movie1, nr_movies):
                if(movie1!=movie2):
                    #gets which users has rated the movies
                    usersRatedMovie1 = rtilde.indices[rtilde.indptr[movie1]: rtilde.indptr[movie1 + 1]]
                    usersRatedMovie2 = rtilde.indices[rtilde.indptr[movie2]: rtilde.indptr[movie2 + 1]]
                    #only interested in movies where there are more than one user
                    users = np.intersect1d(usersRatedMovie1,usersRatedMovie2)
                    numerator = 0
                    den1 = 0
                    den2 = 0
                    ratings1 = rtilde.getcol(movie1)
                    ratings2 = rtilde.getcol(movie2)
                    for user in users:
                        r1 = ratings1[user].data[0]
                        r2 = ratings2[user].data[0]
                        numerator += r1*r2
                        den1 += r1**2
                        den2 += r2**2
                    denominator = (den1*den2)**(1/2)
                    colD.append(movie1)
                    rowD.append(movie2)
                    dij = numerator/(denominator)
                    dataD.append(dij)
        
        D = sp.csr_matrix((dataD,(rowD,colD)),shape=(nr_movies,nr_movies))
        D = D+D.transpose()
        print(D.toarray())



        
    
    
    def test(self):
        u = self.test_data[:,0]
        m = self.test_data[:,1]
        r = self.test_data[:,2]
        r_average = r.sum()/r.size
        rmatrix = getR(self.test_data).toarray()
        pairs = list(zip(u,m))
        self.test_rhat = self.predict(r_average, pairs)
        self.test_RMSE, self.test_abs_errors = getRMSE(pairs, rmatrix, self.test_rhat)

    def train(self):
        u = self.training_data[:,0]
        m = self.training_data[:,1]
        r = self.training_data[:,2]

        pairs = list(zip(u,m))
        r_average = r.sum()/r.size
        rmatrix = getR(self.training_data).toarray()

        A = getA(self.training_data)
        c = r-r_average
        At = A.transpose()
        b = np.linalg.lstsq((At@A).toarray(), At@c)[0]
        self.bu = b[:nr_users]
        self.bm = b[nr_users:]

        self.train_rhat = self.predict(r_average, pairs)
        self.train_rtile = self.train_rhat 
        self.train_RMSE, self.train_abs_errors = getRMSE(pairs, rmatrix, self.train_rhat)
        

if __name__ == '__main__':
    BLP = NeighborhoodPredictor()

    BLP.train()

    print("Training RMSE: ", BLP.train_RMSE.round(3))

    #BLP.test()

    #print("Test RMSE: ", BLP.test_RMSE.round(3))

