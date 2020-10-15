import numpy as np


class SVM(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=1):
        """
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of SVM f(x) = w^T + b

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
        #fx = []
        var = np.matmul(self.w, np.transpose(x)) + self.b
        return var


    def loss(self, x, y):
        """
        Return the SVM hinge loss
        L(x) = max(0, 1-y(f(x))) + 0.5 w^T w

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
        var1 = (1 - y*(self.forward(x)))
        var1_max = np.maximum(0,var1)
        var2 = np.mean(var1_max) +  0.5*(self.l2_reg*np.matmul(self.w,np.transpose(self.w)))[0][0]
        return(var2)



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
        n, = y.shape
        var1 = 1 - y*(self.forward(x))
        var2 = np.zeros(n)
        for i in range(n) :
            if var1[0,i] >= 0 :
                var2[i] = -y[i]
            else :
                var2[i] = 0
        return(np.mean(var2))

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

        var1 = 1 - y * (self.forward(x))
        n, = y.shape

        var2 = np.zeros((x.shape[0],x.shape[1]))
        for i in range(x.shape[0]):
            if var1[0, i] >= 0:
                var2[i] = -y[i]*x[i]
            else:
                var2[i] = 0
        var2 = np.mean(var2,0)
        return var2 + self.l2_reg*self.w


    def fit(self, x, y, plot=False):
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
        self.w = np.random.rand(1, x.shape[1])
        self.b = 0
        loss1 = np.zeros(self.n_epochs)
        for i in range(self.n_epochs):
            self.b = self.b - self.lr * self.grad_loss_wrt_b(x, y)
            self.w = self.w - self.lr * self.grad_loss_wrt_w(x, y)
            loss1[i] = self.loss(x, y)
        return loss1

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
        y = self.forward(x)
        (n, m) = y.shape
        pred = np.zeros(m)
        for i in range(m):
            if y[0, i] >= 0:
                pred[i] = 1
            else:
                pred[i] = -1
        return pred