import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        self.x = np.copy(x)
        #self.locs = []
        n_s = x.shape[0]
        i_c = x.shape[1]
        n_r = x.shape[2]
        n_c = x.shape[3]
        stride = self.size
        r1 = np.zeros((n_s,i_c,((n_r-self.size)/stride)+1,((n_c-self.size)/stride)+1))
        #self.locs = np.zeros(x.shape)
        self.locs = np.zeros((n_s, i_c, ((n_r - self.size) / stride) + 1, ((n_c - self.size) / stride) + 1))
        out_max = np.zeros((n_s, i_c, ((n_r - self.size) / stride) + 1, ((n_c - self.size) / stride) + 1))
        '''
        for l in range(n_s):
            for k in range(i_c):
                i1=0
                for i in range(0,n_r,self.size):
                    j1=0
                    for j in range(0,n_c,self.size):
                        self.locs[l,k,i1,j1] = np.argmax(x[l,k,i:i+self.size,j:j+self.size])

                        j1 = j1+1
                    i1 = i1+1
        '''
        for l in range(n_s):
            for i in range(1+(n_r-self.size)/stride):
                for j in range(1+(n_c-self.size)/stride):
                    i1 = i * stride
                    i2 = i * stride + self.size
                    j1 = j * stride
                    j2 = j * stride + self.size
                    window = self.x[l, :, i1:i2, j1:j2]
                    out_max[l,:,i,j] = np.max(window.reshape((i_c,self.size*self.size)),axis =1)
                    self.locs[l,:,i,j] = np.argmax(window.reshape((i_c,self.size*self.size)),axis =1)

        return out_max
        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        n_s = self.x.shape[0]
        i_c = self.x.shape[1]
        n_r = self.x.shape[2]
        n_c = self.x.shape[3]
        stride=np.copy(self.size)
        x_grad = np.zeros(self.x.shape)
        for l in range(n_s):
            for k in range(i_c):
                for i in range(1+(n_r-self.size)/stride):
                    for j in range(1+(n_c-self.size)/stride):
                        i1 = i*stride
                        i2 = i*stride + self.size
                        j1 = j*stride
                        j2 = j*stride + self.size
                        window  = self.x[l,k,i1:i2,j1:j2]
                        window2 = np.reshape(window,(self.size*self.size))
                        window3 = np.zeros(window2.shape)
                        a1 = np.where(window2 == np.max(window2))
                        #window3[np.argmax(window2)] = 1
                        window3[a1] = 1

                        x_grad[l,k,i1:i2,j1:j2] = np.reshape(window3,(self.size,self.size)).astype('float64') * y_grad[l,k,i,j].astype('float64')
        return x_grad

        #raise NotImplementedError

    def update_param(self, lr):
        pass


