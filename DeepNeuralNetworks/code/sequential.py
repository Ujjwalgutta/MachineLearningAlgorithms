from __future__ import print_function
import numpy as np



class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss


    def forward(self, x, target):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        self.x = np.copy(x)
        k11 = np.copy(self.x)
        #self.target = np.copy(target)


        for layer in self.layers:
            k11 = layer.forward(k11)

        if target is not None:
            a11 = self.loss.forward(k11,target)
            return a11
        else:
            return k11
        #return a11

        #raise NotImplementedError

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        k12 = self.loss.backward()
        for layer in reversed(self.layers):
            k12 = layer.backward(k12)
        return k12
        #raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        self.lr =np.copy(lr)
        for layer in self.layers:
            params = layer.update_param(self.lr)



        #raise NotImplementedError

    def fit(self, x, y, epochs=20, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.epochs = np.copy(epochs)
        self.lr = np.copy(lr)
        self.batch_size = np.copy(batch_size)
        l1 = np.zeros((1,self.epochs))

        for j in range(epochs):
            for i in range(0,self.x.shape[0],self.batch_size):
                r1 = self.forward(self.x[i:i+self.batch_size,:],self.y[i:i+self.batch_size,:])
                self.backward()
                self.update_param(self.lr)
                #r2 = self.loss.forward(r1)
                l1[j] = r1


            #z1 = self.update_param(self.lr)
        return l1


        #raise NotImplementedError

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        #self.x = np.copy(x)

        return np.argmax(self.forward(x),axis = 1)
        #raise NotImplementedError
