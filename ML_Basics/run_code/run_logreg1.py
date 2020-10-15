from logistic_regression import LogisticRegression
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)

model1 = LogisticRegression(n_epochs=100,lr=0.1,l2_reg=0)
loss1 = model1.fit(x_train,y_train)

model2 = LogisticRegression(n_epochs=100,lr=0.01,l2_reg=0)
loss2 = model2.fit(x_train,y_train)

model3 = LogisticRegression(n_epochs=100,lr=50.0,l2_reg=0)
loss3 = model3.fit(x_train,y_train)

plt.figure(1)
plt.plot(range(1,101) , loss1, 'C0' , label = 'lr=0.1')
plt.plot(range(1,101) , loss2, 'C1' , label = 'lr=0.01')
plt.plot(range(1,101) , loss3, 'C2' , label = 'lr=50.0')
plt.ylabel('loss function value')
plt.xlabel('epochs')
plt.legend()
plt.show()

