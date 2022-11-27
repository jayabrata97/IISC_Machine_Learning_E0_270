# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:13:25 2020
Q3b of E0-270 assignment
@author: Jayabrata
"""
import numpy as np
import matplotlib.pyplot as plt

## Training data
no_samples = 100
x_min = [-1]
x_max = [1]
x = np.random.uniform(low = x_min, high = x_max, size = (no_samples,1))
#print(x)
y = []
for i in range(no_samples):
    y.append(np.sin(3*x[i]).tolist())

## Finding optimal value of w , setting b=0
for j in range(no_samples):
    w = np.sum(y[j]*x[j]) / np.sum((x[j])**2)
    
## prediction
y_pred = []
for j in range(no_samples):
    y_pred.append(w*x[j])

## Mean square error
MSE = 0
error = 0
for k in range(no_samples):
    error += (y[k] - y_pred[k])**2
MSE = error / (no_samples)
print("w is: ",w)
print("Mean Square Error on training data: ",MSE)    

    
## plotting 
plt.scatter(x, y, c='blue', marker='x')    
plt.plot(x, w*x, c='red')
plt.xlabel('x')
plt.ylabel("y")
plt.title("Plot of Q3b, blue= data, red= linear regression line")
plt.grid()
plt.show()
