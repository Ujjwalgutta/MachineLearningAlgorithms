import numpy as np


class LeastSquares(object):
    def __init__(self, k = 1):
        """
        Initialize the LeastSquares class

        The input parameter k specifies the degree of the polynomial
        """
        self.k = k
        self.coeff = None

    def fit(self, x, y):
        """
        Find coefficients of polynomial that predicts y given x with
        degree self.k

        Store the coefficients in self.coeff
        """
        n = x.shape[0]
        A = np.zeros((n,self.k+1))
        for i in range (n):
            for j in range (0,self.k+1):
                A[i,j] = x[i] ** j

        A_inv = np.linalg.pinv(A)
        self.coeff = np.matmul(A_inv,y)
        #raise NotImplementedError

    def predict(self, x):
        """
        Predict the output given x using the learned coeffecients in
        self.coeff
        """
        n = x.shape[0]
        A = np.zeros((n, self.k + 1))
        for i in range(n):
            for j in range(0, self.k + 1):
                A[i, j] = x[i] ** j
        x = np.matmul(A,self.coeff)
        #print(x)
        return x
        #raise NotImplementedError
