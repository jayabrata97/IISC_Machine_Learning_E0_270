# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:13:29 2020
Q3d of E0-270 assignment
@author: Jayabrata
"""
import numpy as np
import matplotlib.pyplot as plt

## Definition of kernel
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

## Training data
no_samples = 100 # have to change here to 100
x_min = [-1]
x_max = [1]
x_train = np.random.uniform(low = x_min, high = x_max, size = (no_samples,1))
#print(x)
y = []
for i in range(no_samples):
    y.append(np.sin(3*x_train[i]).tolist())
#print(y)

## Kernel matrix
K = []
for j in range(no_samples):
    for k in range(no_samples):
      K.append(linear_kernel(x_train[j], x_train[k]))     
K = np.reshape(K, (no_samples, -1))
#print(K)
I = np.eye(no_samples)

## Prediction on training set
y_train = []
lamb = float(input("Enter regularization coefficient:"))
#lamb=1.0
inv = np.matmul((np.linalg.inv(K + (lamb * I))),y)
#print(inv)   
k_train = []
for m in range(no_samples):
    k_train.append(linear_kernel(x_train,x_train[m]))
#print(k_train)
for l in range(no_samples):
    y_train.append(np.dot(k_train[l],inv))
#print(y_train)    
error = 0
for n in range(no_samples):
    error += (y[n] - y_train[n])**2
MSE = error / (no_samples) *100
print("Mean Square Error for training data in percentage : ",MSE)    
    


## Defining test data
no_test = 90  
x_min = [-1]
x_max = [1]
x_test = np.random.uniform(low = x_min, high = x_max, size = (no_test,1))
y_test = []
for p in range(no_test):
    y_test.append(np.sin(3*x_test[p]).tolist())
#print(y)
        

y_pred = []   
k_test = []
for r in range(no_test):
    k_test.append(linear_kernel(x_train,x_test[r]))
#print(k_train)
for s in range(no_test):
    y_pred.append(np.dot(k_test[s],inv))
#print(y_pred)    
error_test = 0
for t in range(no_test):
    error_test += (y_test[t] - y_pred[t])**2
MSE_test = error / (no_test) *100
print("Mean Square Error for testing data in percentage : ",MSE_test)         
        
plt.scatter(x_train, y,c='blue', marker='x')    
plt.plot(x_train, y_train, c='red')
plt.xlabel('x')
plt.ylabel("y")
plt.title("Plot of Q3d, blue= data, red= learned function")
plt.grid()
plt.show()        