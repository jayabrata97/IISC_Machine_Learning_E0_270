# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:59:31 2020
helper    q4b
@author: User
"""

import numpy as np
import dill

filename = "kernel_4a.pkl"
input_file = open(filename,"rb")
kernel_func = dill.loads(dill.load(input_file))

## Defining the distance function  
def dist2(X, Y, m, k):
    dist =[]
    for a in range(m):
        mid_term1=0
        for i in range(m):
            mid_term1 += k(X[i],Y[i])
        mid_term = (2/m)*mid_term1
        last_term1 = 0
        for j in range(m):
            for n in range(m):       # changed from iter count k to n because of call error
                last_term1 += k(Y[j],Y[n]) 
        last_term = (1/(m**2))*last_term1    
        dist.append((k(X[i],X[i]) - mid_term + last_term)**0.5) 
    dist2 = 0
    for b in range(m):
        dist2 += dist[b]
    return dist2
 
d = 10 # have to change here to 10
I = np.eye(d)
#print(I)

#print(kernel_func(I[0],I[2]))
sum =dist2(I,I,d,kernel_func)
#for i in range(d):
#    D.append(dist2(I[i], I, d, kernel_func))
#print(D)

## Summing all elements of D 
#sum =[]
#for j in range(len(D)):
#       sum += D[j]
       
print("Final sum of d_i's  in Q4b is: ",sum)

       
    