import numpy as np
from scipy import stats


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """

        self.x_train = np.copy(x)
        self.y_train = np.copy(y)
        #raise NotImplementedError

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """
        (N,D) = x.shape
        euc_dist = np.zeros((N,1))
        final_euc_dist = np.zeros((N,N))

        #temp = np.zeros((N,1))
        #dist = np.zeros((0,N))
        self.x_test = np.copy(x)
        for i in range (N):
            dist = ((self.x_test[i] - self.x_train) **2 )
            for j in range(N):
                euc_dist[j] = (dist[j][0] + dist[j][1])**0.5

            final_euc_dist[:,[i]] = euc_dist


        temp = np.zeros((N,N))
        for i in range(N):
            temp[:,[i]] = np.argsort(final_euc_dist[:,[i]],axis=0)

        temp = temp.astype(np.int)
        y2 = self.y_train
        #r = y2.shape
        y1 = np.zeros((N,N))
        for i in range (N):
            for j in range (N):
                y1[i,j] = y2[temp[i][j]]

        k1 =self.k
        y3 = np.zeros((k1,N))
        y1 = y1.astype(np.int)
        for i in range(k1):
            for j in range(N):
                y3[i][j] = y1[i][j]

        y4 = stats.mode(y3,axis=0)
        y5 =y4[0]
        y6 = y5[0,:]



        return y6
        #raise NotImplementedError
