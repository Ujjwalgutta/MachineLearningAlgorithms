import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.n_i = np.copy(n_i)
        self.n_o = np.copy(n_o)
        #np.random.seed(30)
        self.W= np.random.normal(0,np.sqrt(2./float((n_i+n_o))),(n_o,n_i))
        #self.W = np.random.rand(self.n_o,self.n_i)
        self.b = np.zeros((1,self.n_o))
        self.x = None
        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        W = self.W
        b = self.b
        #(N,D) =  x.shape()
        self.x = np.copy(x)
        f_full = np.dot(self.x,np.transpose(self.W))  + self.b
        return f_full


        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        #self.y_grad = np.copy(y_grad)
        #f_x = self.W
        #f_b = np.identity(self.n_o)
        #self.b_grad = np.zeros((1,self.n_o))
        #self.W_grad = np.zeros((self.n_o,self.n_i))
        l_x = np.dot(y_grad,self.W)
        y_gradt = np.transpose(y_grad)
        self.b_grad = np.sum(y_grad, axis=0, keepdims=True)
        self.W_grad = np.dot(y_gradt,self.x)

        return l_x
        #raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.lr = np.copy(lr)
        self.b = self.b - (self.lr*self.b_grad)
        self.W = self.W - (self.lr*self.W_grad)

        #raise NotImplementedError
