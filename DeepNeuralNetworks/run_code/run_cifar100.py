"""
=> Your Name:

1. In this script, You need to implement the simple neural network using code presented in Section 4.1.
2. Using the network above, plot the average training loss vs epoch for learning rates 0.1, 0.01, 0.3 for 20 epochs.
3. Run the same network with learning rate 10 and observe the result.
4. Report the test accuracy with learning rates 0.1, 0.01, 0.3 and 10 for 20 epochs.

=> After running this script, describe below what you observe when using learning rate 10:


"""
import numpy as np
import matplotlib.pyplot as plt

from layers import (FullLayer,ReluLayer, SoftMaxLayer,CrossEntropyLayer, Sequential)
from layers.dataset import cifar100
model = Sequential(layers = (FullLayer(32*32*3,500),ReluLayer(),FullLayer(500,5),SoftMaxLayer()),loss = CrossEntropyLayer())
(x_train,y_train) , (x_test,y_test) = cifar100(1215350903)

k1 = model.fit(x_train,y_train,epochs=20,lr=0.1,batch_size=128)
k2 = model.fit(x_train,y_train,epochs=20,lr=0.01,batch_size=128)
k3 = model.fit(x_train,y_train,epochs=20,lr=0.3,batch_size=128)
k4 = model.fit(x_train,y_train,epochs=20,lr=10,batch_size=128)


plt.figure(1)
plt.plot(range(1,21) , k1, 'C0' , label = 'lr=0.1')
plt.plot(range(1,21) , k2, 'C1' , label = 'lr=0.01')
plt.plot(range(1,21) , k3, 'C3' , label = 'lr=0.3')
plt.plot(range(1,21) , k3, 'C4' , label = 'lr=10')

plt.ylabel('loss function value')
plt.xlabel('epochs')
plt.legend()
plt.show()



