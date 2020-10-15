"""
Run least squares with provided data
"""

import numpy as np
import matplotlib.pyplot as plt
from ls import LeastSquares
import pickle

# load data
data = pickle.load(open("ls_data.pkl", "rb"))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

ls =LeastSquares(2)
ls.fit(x_train,y_train)
pred_test = ls.predict(x_test)
pred_train = ls.predict(x_train)

N=20
M = np.arange(1,21,1)
MSE_train = np.zeros(20)
MSE_test = np.zeros(20)
# try ls
#ls = list()
for i in range(1, N+1):
    #print(i)
    obj = (LeastSquares(i))
    obj.fit(x_train, y_train)
    train_temp = obj.predict(x_train)
    test_temp = obj.predict(x_test)
    MSE_train[i-1] = ((train_temp - y_train) ** 2).mean()
    MSE_test[i-1] = ((test_temp - y_test) ** 2).mean()

#print("TRAIN", MSE_train)
#print("TEST", MSE_test)






plt.figure(1)
plt.plot(M,MSE_train  , 'r*', label='Train Error')
plt.plot(M,MSE_test  ,'y*', label='Test Error')
plt.legend()
plt.figure(2)
plt.plot(x_test, pred_test, 'r*', label='Predicted')
plt.plot(x_test, y_test, 'y*', label='Ground truth')
plt.legend()
plt.show()
