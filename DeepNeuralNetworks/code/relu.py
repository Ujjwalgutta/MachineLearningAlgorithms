import numpy as np


class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

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
             The output data (need to store for backwards pass)
        """
        self.x = np.copy(x)
        (h,g) = self.x.shape
        self.y = np.zeros(x.shape)
        for i in range (h):
            for j in range(g):
                if self.x[i][j] > 0:
                    self.y[i][j]  = self.x[i][j]
                else:
                    self.y[i][j] = 0
        return self.y
        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        l_y = np.zeros(y_grad.shape)
        self.y_grad  = y_grad
        (N,D) = self.y.shape
        for i in range (N):
            for j in range(D):
                if self.y[i][j] > 0:
                    l_y[i][j] = 1
                else:
                    l_y[i][j] = 0
        return y_grad*l_y



        #raise NotImplementedError

    def update_param(self, lr):
        pass  # no parameters to update
