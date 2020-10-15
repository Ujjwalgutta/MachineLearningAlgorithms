from knn import KNN
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=800, n_test=800)

#model = KNN(k=3)
#model.fit(x_train, y_train)

N = y_test.shape[0]
#y_pred  = np.zeros((N,51))
acc1 = []
for i in range(1,51,5):
    model = KNN(k=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc = np.mean(y_pred == y_test)
    acc1.append(acc)

acc2 = np.asarray(acc1)
SH = acc2.shape
k_val  = np.arange(1,51,5)
plt.figure()
plt.plot(k_val,acc2)
plt.title("Accuracy Plot")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.show()
