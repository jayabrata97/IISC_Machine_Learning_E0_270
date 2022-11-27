# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:20:11 2020
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
x0 = np.ones((no_samples, 1))
x_tilda = np.hstack((x0,x))


## finding optimal value of w and b
x_transpose = x_tilda.T  
a = np.linalg.inv(np.dot(x_transpose, x_tilda))   
b = np.dot(x_transpose , y)
w_tilda = np.dot(a, b)  
#print("w_tilda is: \n",w_tilda)
w = w_tilda[1]
b = w_tilda[0]
print("w is: ",w)
print("b is: ",b)
 
## prediction
y_pred = []
for j in range(no_samples):
    y_pred.append(w*x[j]+b)
    
## Mean square error
error = 0
for k in range(no_samples):
    error += (y[k] - y_pred[k])**2
MSE = error / (no_samples)

print("Mean Square Error on training data: ",MSE)     
    
plt.scatter(x, y,c='blue', marker='x')    
plt.plot(x, w*x+b, c='red')
plt.xlabel('x')
plt.ylabel("y")
plt.title("Plot of Q3b, blue= data, red= linear regression line")
plt.grid()
plt.show()  
  