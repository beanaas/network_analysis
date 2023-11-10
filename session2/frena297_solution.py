import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
import datetime

filename = "albintest"
nr_users = 2000
nr_movies = 1500

nr_users = 10
nr_movies = 5
u_min = 0
L = 2

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
        self.train_rtilde = None
        self.D = None
        self.rm = None
        self.rated_movies = None
        
    def getD(self, r_average, pairs):
        rmatrix = getR(self.training_data)
        self.rm = rmatrix
        start = datetime.datetime.now()
        ci = []
        ri =[]
        data = []
        #key is user, val is index to all movies rated
        rated_movies = {}
        #key is movie, val is index to all users that rated this movie
        user_rated = {}
        for user, movie in pairs:
            val = (r_average+self.bu[user]+self.bm[movie])

            if(user in rated_movies):
                rated_movies[user].append(movie)
            else:
                rated_movies[user] = [movie]

            if(movie in user_rated):
                user_rated[movie].append(user)
            else:
                user_rated[movie] = [user]

            if(val>5):
                val = 5
            elif(val<1):
                val = 1
            ri.append(user)
            ci.append(movie)
            data.append(val)
        self.rated_movies = rated_movies
        print("first",datetime.datetime.now()-start)

        rhat = sp.csr_matrix((data,(ri,ci)),shape=(nr_users,nr_movies))

        rtilde = (rmatrix-rhat).tocsc()
        self.train_rtilde = rtilde
        rowD = []
        colD = []
        dataD = []

        print("before loop")
        start = datetime.datetime.now()
        
        non_zeros = {}
        for movie1 in range(nr_movies):
            r1 = rtilde.getcol(movie1).toarray().flatten()
            non_zeros[movie1] = (np.nonzero(r1),r1)


        for movie1 in range(nr_movies):
            movie1_props = non_zeros[movie1]
            r1 = movie1_props[1]
            usersRatedMovie1 = movie1_props[0]
            for movie2 in range(movie1, nr_movies):
                if(movie1!=movie2):
                    movie2_props = non_zeros[movie2]
                    r2 = movie2_props[1]
                    usersRatedMovie2 = movie2_props[0]
                    users = np.intersect1d(usersRatedMovie1,usersRatedMovie2)
                    if(len(users)>=u_min):
                        
                        numerator = r1.transpose().dot(r2)
                        #denominator = np.sqrt(np.sum(np.square(r1[users])) * np.sum(np.square(r2[users])))
                        denominator = np.linalg.norm(r1[users])*np.linalg.norm(r2[users])
                        dij = numerator/(denominator)
                        if(dij):
                            colD.append(movie1)
                            rowD.append(movie2)
                            dataD.append(dij)

        print(datetime.datetime.now()-start)
        D = sp.csc_matrix((dataD,(rowD,colD)),shape=(nr_movies,nr_movies))
        D = D+D.transpose()
        
        #diagonal_indices = range(min(D.shape[0], D.shape[1]))
        #D[diagonal_indices, diagonal_indices] = 1
        print("D array")
        print(D.toarray()[:5, :5])

        self.D = D

    def predict(self, r_average, pairs):
        
        LD = {}
        for movie1 in range(nr_movies):
            sim_col =self.D.getcol(movie1).data
            abs_sim_col = np.abs(sim_col)
            sorted_simcol = np.argsort(-abs_sim_col)
            LD[movie1] = sorted_simcol

        ci = []
        ri =[]
        #index of sorted largest indexes
        data = []

        print("before loop")
        start = datetime.datetime.now()
        rtilde = self.train_rtilde.tocsr()
        print("last",datetime.datetime.now()-start)
        start = datetime.datetime.now()
        rt = rtilde.toarray()
        dm = self.D.toarray()

        for user, movie in pairs:
            val = (r_average+self.bu[user]+self.bm[movie])
            numerator = []
            denominator = []

            sorted_list = LD[movie].data
            numerator = []
            denominator = []
            for i in sorted_list[:L]:
                di = sorted_list[i]
                if(di in self.rated_movies[user]):
                    d = dm[movie][di]
                    r = rt[user][di]
                    numerator.append(d*r)
                    denominator.append(np.abs(d))

            denominator = np.sum(denominator)
            numerator = np.sum(numerator)

            """most_similar_movies_indexes = sorted_list[:L]
            new_most_similar_movies_indexes = []
            for i in most_similar_movies_indexes:
                new_most_similar_movies_indexes.append(i)
            new_most_similar_movies_indexes = [x+1 for x in new_most_similar_movies_indexes]
            most_similar_movies_d = self.D.getcol(movie).data[most_similar_movies_indexes]
            #need to cast it toarray for using indexes
            rtilde_uj = rtilde.getrow(user).toarray()[0][most_similar_movies_indexes]
            numerator = most_similar_movies_d.dot(rtilde_uj)
            denominator = np.sum(np.abs(most_similar_movies_d))
            print(user, movie)"""
            if(denominator>0):
                #print("before neighbour: ", val)
                val = val + (numerator/denominator)
                #print("after neighbour: ", val)
                #print("real: ", self.rm.toarray()[user][movie])
            if(val>5):
                val = 5
            elif(val<1):
                val = 1
            ri.append(user)
            ci.append(movie)
            data.append(round(val, 2))

        print("last",datetime.datetime.now()-start)

        rhatnew = sp.csr_matrix((data,(ri,ci)),shape=(nr_users,nr_movies))
        print(rhatnew.toarray())
        print("DONE rhatnew")

        return rhatnew.toarray()
    
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

        self.getD(r_average, pairs)
        self.train_rhat = self.predict(r_average, pairs)
        self.train_RMSE, self.train_abs_errors = getRMSE(pairs, rmatrix, self.train_rhat)
        

if __name__ == '__main__':
    start = datetime.datetime.now()
    NHP = NeighborhoodPredictor()

    NHP.train()

    print("NHP Training RMSE: ", NHP.train_RMSE.round(3))

    NHP.test()

    print("NHP Test RMSE: ", NHP.test_RMSE.round(3))
    print("tot time: ", datetime.datetime.now()-start)

"""    BLP = BaseLinePredictor()

    BLP.train()

    print("BLP Training RMSE: ", BLP.train_RMSE.round(3))

    BLP.test()
    

    print("BLP Test RMSE: ", BLP.test_RMSE.round(3))"""
    



