"""
=> Your Name: Ujjwal Gutta

In this script, you need to plot the average training loss vs epoch using a learning rate of 0.1 and a batch size of 128 for 15 epochs.

=> Final accuracy on the test set: 83.4%

"""
import numpy as np
import matplotlib.pyplot as plt
from layers import (ConvLayer,MaxPoolLayer,FlattenLayer, FullLayer,ReluLayer, SoftMaxLayer,Sequential,CrossEntropyLayer)
from layers.dataset import cifar100
model = Sequential(layers = (ConvLayer(3,16,3),ReluLayer(),MaxPoolLayer(2),ConvLayer(3,32,3),ReluLayer(),MaxPoolLayer(2),FlattenLayer(),FullLayer(2048,3),SoftMaxLayer()),loss = CrossEntropyLayer())
(x_train,y_train) , (x_test,y_test) = cifar100(1215350903)

k1 = model.fit(x_train,y_train,epochs=15,lr=0.1,batch_size=128)


plt.figure(1)
plt.plot(range(1,16) , k1, 'C0' , label = 'lr=0.1')
plt.ylabel('loss function value')
plt.xlabel('epochs')
plt.legend()
plt.show()
y_pred = model.predict(x_test)
print("knn accuracy: "+ str(np.mean(y_pred == y_test)))



