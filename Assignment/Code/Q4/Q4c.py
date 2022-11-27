# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:03:30 2020
not completed, Q4c
@author: User
"""

import dill
import copy
import numpy as np
import matplotlib.pyplot as plt


filename1 = "kernel_4a.pkl"
input_file1 = open(filename1,"rb")
kernel_func = dill.loads(dill.load(input_file1))

filename2 = "data.npy"
data = np.load(filename2)

plt.scatter(data[:,0],data[:,1],marker='x')

## Initializing means
K = 2
x1x2_min = [-4, -4]
x1x2_max = [4, 4]
mu = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (K,2))
print('initial mu are:\n',mu)
plt.scatter(mu[:,0],mu[:,1],marker='x')
plt.show()

## distance function
def dist2(X, Y, m, k):
    dist2 =[]
    for a in range(m):
        dist2.append(k(X[a],X[a])-2*k(X[a],Y[a])+k(Y[a],Y[a]))
    error = 0
    for b in range(m):
        error += dist2[b]
        error = error**0.5
    return error


def dist1(x, Y, k):  # it should return a vector of distances between x and each Y[i] 
    dist1 = []
    for i in range(C.shape[0]):
        dist1.append((k(x,x) - 2*k(x,Y[i]) +k(Y[i],Y[i])))
    return dist1
  
## k means algo
C_old = np.zeros(mu.shape)   # changed here from len() data to mu.shape 
clusters = np.zeros(len(data))
C = mu
#error = dist2(C, C_old, mu.shape[0] , kernel_func)   # have to think here 
#print("Initial error is:\n",error)


while (C!=C_old).all():
    for i in range(len(data)):
        distances = dist1(data[i], C, kernel_func)
        cluster = np.argmin(distances)
        clusters[i] = cluster
        #print('cluster is\n',clusters)
    C_old = copy.deepcopy(C)
    for i in range(K):
        points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])  
        C[i] = np.mean(points, axis=0)
    #error = dist2(C, C_old, 2, kernel_func)   
    print("if C!=C_old",C!=C_old.all())
#print("Final C is:\n",C) 
        
#C = mu
#error = dist2()



