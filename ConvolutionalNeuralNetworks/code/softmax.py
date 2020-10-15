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
