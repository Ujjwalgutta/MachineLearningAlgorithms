import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None


    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        self.x = np.copy(x)
        x_max = np.max(self.x)
        y1 = self.x - x_max
        #print("matrix is",self.y)
        (N,D) = y1.shape
        self.y = np.zeros(self.x.shape)
        for i in range(N):
            for j in range(D):
                self.y[i][j] = (np.exp(y1[i][j])) / (np.sum(np.exp(y1[i,:])))

        #for i in range(2):
         #   l1 = np.sum(np.exp(y1[i,:]))

        return self.y

        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        """

        self.y_grad = np.copy(y_grad)
        (a,b) = self.y.shape
        y1t = np.copy(self.y)
        #for i in range(a):
        a1 = np.diag(y1t[0,:])
        a2 = np.diag(y1t[1,:])
        z_y1 = y1t[0:1,:].T.dot(y1t[0:1,:])
        z_y2 = y1t[1:2,:].T.dot(y1t[1:2,:])
        f1 = a1 - z_y1
        f2 = a2 - z_y2
        k1 = self.y_grad[0:1,:].dot(f1)
        k2 = self.y_grad[1:2,:].dot(f2)
        z_y3 = np.vstack((k1,k2))


        return z_y3
        """
        n = np.shape(self.y)
        grad_input = np.zeros(self.y.shape)
        for i in range(n[0]):
            j = np.diag(self.y[i,:]) - np.dot(self.y[i:i+1,:].T,self.y[i:i+1,:])
            grad_temp = np.dot(y_grad[i:i+1,:],j)
            grad_input[i] = grad_temp
        np.vstack((grad_input,grad_temp))
        return grad_input
        #raise NotImplementedError


    def update_param(self, lr):
        pass  # no learning for softmax layer
