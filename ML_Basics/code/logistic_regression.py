import numpy as np


class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        """
        Initialize variables
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of logistic regression

        This will return the squashing function:
        f(x) = 1 / (1 + exp(-(w^T x + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        (N,D) = x.shape
        #self.w = np.random.rand(1,D)
        w = self.w

        self.x = np.copy(x)
        b = self.b
        f = 1 / (1 + np.exp(-(np.dot(w, np.transpose(self.x)) + b)))
        return f

        #raise NotImplementedError

    def loss(self, x, y):
        """
        Return the logistic loss
        L(x) = 1/N * (ln(1 + exp(-y * (w^Tx + b)))) + 1/2 * lambda * w^T * w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        (N,D) = x.shape
        k1 = np.matmul(x,np.transpose(self.w)) + self.b
        y1 = y.reshape((N,1))
        c2 = 0
        c1 = (np.log(1+np.exp(-1*y1*k1)))
        for i in range(N):
            c2 += c1[i][0]
        l = c2 / N + (0.5 * self.l2_reg * np.dot(self.w,np.transpose(self.w)))
        l1 = l[0][0]
        return l1


        #raise NotImplementedError


    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """
        (N,D) = x.shape
        k1 = np.matmul(x,np.transpose(self.w)) + self.b
        y1=y.reshape((N,1))
        dr = (1+np.exp(1*y1*k1))
        nr = -y1
        c2=0
        c1 = nr/dr
        for i in range(N):
            c2 +=c1[i][0]
        l_b = c2 / N
        #b2 = np.copy(self.b)
        #b1 = np.zeros((10,1))
        #b1[0] = b2
        #for i in range(1,10):
            #b1[i] = b1[i-1] - self.lr*l_b



        return l_b


        #raise NotImplementedError

    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """
        (N, D) = x.shape
        k1 = np.matmul(x, np.transpose(self.w)) + self.b
        y1 = y.reshape((N,1))
        dr = (1 + np.exp(1 * y1 * k1))
        nr = -y1 * x
        c1 = nr/dr
        #(N1,D1)  = self.w.shape
        #c2 = np.zeros((N1,D1))
        #for i in range(N):
        #    c2[i-1] = c1[i-1,:] + c1[i,:]
        #l_w = c2/N
        l_w1 = np.mean(c1,axis=0)
        return l_w1


        #raise NotImplementedError

    def fit(self, x, y):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.w = np.random.rand(1,x.shape[1])
        self.b = 0
        loss = []
        for i in range(self.n_epochs):
            self.b = self.b - (self.lr * self.grad_loss_wrt_b(x,y))
            self.w = self.w - (self.lr * self.grad_loss_wrt_w(x,y))
            loss.append(self.loss(x,y))
        return loss



    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """
        (N,D) = x.shape
        k1 = np.matmul(x, np.transpose(self.w)) + self.b
        dr = (1 + np.exp(-1*k1))
        nr = 1.0
        f_x = nr / dr
        f_x1  = f_x.reshape((N,))
        y1 = np.zeros(N)
        for i in range(N):
            if f_x1[i] > 0.5:
                y1[i] = 1
            else:
                y1[i] = -1


        y2 = y1.astype(int)
        return(y2)


        #raise NotImplementedError
