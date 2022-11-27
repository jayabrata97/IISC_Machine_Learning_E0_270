# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:56:19 2020
q3c, check line 31 and 37
@author: Joy
"""
import numpy as np
import matplotlib.pyplot as plt

def phi_func(x, no_samples,fea_dim):
    phi = []
    for i in range(no_samples):
        for j in range(fea_dim): 
            phi.append((x[i]**(j+1)))       
    phi = np.reshape(phi,(no_samples,-1))  
    
    return phi

def phi_tild(x, no_samples,fea_dim):
    phi = []
    for i in range(no_samples):
        for j in range(fea_dim): 
            phi.append((x[i]**(j+1)))       
    phi = np.reshape(phi,(no_samples,-1))        
    x0 = np.ones((no_samples, 1))
    phi_tild = np.hstack((x0,phi))    
    
    return phi_tild

def y_data(x, no_samples):
    y= []
    for k in range(no_samples):
        y.append(np.sin(3*x[k]).tolist()) 
     
    return y    

## Training data
no_samples = 100  # it should be changed to 100
x_min = [-1]
x_max = [1]
x = np.random.uniform(low = x_min, high = x_max, size = (no_samples,1))
#print(x)

fea_dim = 10   # have to change to 10 here 
for i in range(fea_dim):
    phi_tilda = phi_tild(x,no_samples,(i+1))
    #print('phi_tilda: \n',phi_tilda)
    phi_transpose = phi_tilda.T    
    #print('phi_transpose: \n',phi_transpose)
    a = np.linalg.inv(np.dot(phi_transpose, phi_tilda)) 
    y = y_data(x, no_samples)
    b = np.dot(phi_transpose , y)
    w_tilda = np.dot(a, b)  
    #print("w_tilda is: \n",w_tilda)
    w = w_tilda[1:]    
    #print('w is: \n',w)
    b = w_tilda[0]
    #print('b is: \n',b) 
    phi = phi_func(x, no_samples, (i+1))
    #print('phi:\n',phi)
    y_pred = []
    for j in range(no_samples):
        y_pred.append(np.matmul(phi[j],w) + b)
    #print('y_pred: \n',y_pred)   
    for k in range(no_samples):
        error= 0
        MSE = 0
        error += (y[k] - y_pred[k])**2
        MSE = error / (no_samples)
    print("MSE for %d feature no is :%f " %((i+1), MSE))    
    
    plt.scatter(x, y, c='blue', marker='v')
    plt.scatter(x, (np.matmul(phi,w)+b), c='red',marker='x')    
    #plt.plot(x, np.matmul((phi,w)+b), c='red')
    plt.xlabel('x')
    plt.ylabel("y_pred")
    plt.title("Plot of Q3c for k = %s, blue= data, red= learned function" %(i+1))
   # plt.legend("%s" %(i+1))
    plt.grid()
    plt.show()
    
    
    