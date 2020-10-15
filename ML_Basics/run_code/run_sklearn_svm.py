import sklearn.svm
import numpy as np
import datasets
import csv
import matplotlib.pyplot as plt

x_train,y_train,x_test,y_test = datasets.moon_dataset(n_train = 800,n_test = 800)

model  =sklearn.svm.SVC(C = 1.0,kernel = 'linear')

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = np.mean(y_pred == y_test)
acc_1 = str.format('{0:.4f}',acc)
print("Accuracy for Linear kernel is",acc_1)

model1 = sklearn.svm.SVC(C = 1.0,kernel = 'poly',degree = 3,gamma = 'scale')
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)
acc1 = np.mean(y_pred1 == y_test)
acc_2 = str.format('{0:.4f}',acc1)
print("Accuracy for poly kernel is",acc_2)

model2 = sklearn.svm.SVC(C = 1000.0,kernel = 'rbf',degree =3,gamma = 'scale')
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
acc2 = np.mean(y_pred2 == y_test)
acc_3 = str.format('{0:.4f}',acc2)
print("Accuracy for rbf kernel",acc_3)

model3 = sklearn.svm.SVC(C = 0.01,kernel = 'sigmoid',gamma='scale')
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
acc3 = np.mean(y_pred3 == y_test)
acc_4 = str.format('{0:.4f}',acc3)
print("Accuracy for sigmoid kernel is",acc_4)

csvData = [['Linear Kernel',acc_1] , ['Poly kernel',acc_2] , ['RBF kernel',acc_3] , ['Sigmoid kernel',acc_4]]

with open('svm_results.csv' , 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()