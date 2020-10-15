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
        (e,f,g,h) = self.x.shape
        self.y = np.zeros(x.shape)
        for i in range (e):
            for j in range(f):
                for k in range(g):
                    for l in range(h):
                        if self.x[i][j][k][l] > 0:
                            self.y[i][j][k][l]  = self.x[i][j][k][l]
                        else:
                            self.y[i][j][k][l] = 0
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
        (n,ic,nr,nc) = self.y.shape
        for i in range (n):
            for j in range(ic):
                for k in range(nr):
                    for l in range(nc):
                        if self.y[i][j][k][l] > 0:
                            l_y[i][j][k][l] = 1
                        else:
                            l_y[i][j][k][l] = 0
        return y_grad*l_y



        #raise NotImplementedError

    def update_param(self, lr):
        pass  # no parameters to update
