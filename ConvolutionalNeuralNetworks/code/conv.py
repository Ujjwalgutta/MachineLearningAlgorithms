import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        #raise NotImplementedError

        self.n_i = n_i
        self.n_o = n_o
        self.h = h
        f_in = n_i*h*h
        f_out = n_o*h*h
        self.W = np.random.normal(0, np.sqrt(2. / float((f_in + f_out))), (n_o, n_i,self.h,self.h))
        self.b = np.zeros((1,self.n_o))
        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutiona

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """

        self.x = np.copy(x)
        a = x.shape[0]
        b = x.shape[2]
        c = x.shape[3]
        x1 = np.pad(x,((0,0), (0,0),((self.h-1)/2,(self.h-1)/2),((self.h-1)/2,(self.h-1)/2) ),'constant')

        l = np.zeros((a,self.n_o,b,c),dtype='float64')
        '''
        for i in range(a):
            for j in range(self.n_o):
                l[i,j] = scipy.signal.correlate(x1[i,:,:,:],self.W[j,:,:,:],mode='valid') + self.b[:,j]

        return l
        '''
        pad_x=(self.h-1)/2
        pad_y = (self.h - 1) / 2
        for i in xrange(0,a):
            temp_inp = x[i,:,:,:]
            inp_padded = np.zeros([self.n_i,b+(2*pad_x),c+(2*pad_y)],dtype='float64')
            for j in xrange(0,self.n_i):
                inp_padded[j,:,:] = np.pad(temp_inp[j,:,:],((pad_x,pad_x),(pad_y,pad_y)),'constant')
            for k in xrange(0,self.n_o):
                l[i,k,:,:] = scipy.signal.correlate(inp_padded,self.W[k,:,:,:],"valid")
                l[i,k,:,:] = l[i,k,:,:] + self.b[0][k]
        return l
        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

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
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        x1 = np.pad(self.x, ((0, 0), (0, 0), ((self.h - 1) / 2, (self.h - 1) / 2), ((self.h - 1) / 2, (self.h - 1) / 2)),'constant')
        a = self.x.shape[0]
        self.b_grad = np.zeros(self.b.shape)
        self.b_grad = self.b_grad + np.sum(y_grad , axis=(0,2,3),keepdims=False)
        x_grad = np.zeros(self.x.shape)
        self.W_grad = np.zeros(self.W.shape)
        for i in range(self.n_o):
            for j in range(self.n_i):
                 self.W_grad[i,j] = scipy.signal.correlate(x1[:,j,:,:],y_grad[:,i,:,:], mode='valid')

        for i in range(a):
            for j in range(self.n_o):
                for k in range(self.n_i):
                    x_grad[i,k] = x_grad[i,k] + scipy.signal.convolve(y_grad[i,j,:,:] , self.W[j,k,:,:], mode = 'same')

        return x_grad
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
        self.b = self.b - (lr*self.b_grad)
        self.W = self.W - (lr*self.W_grad)
        #raise NotImplementedError
