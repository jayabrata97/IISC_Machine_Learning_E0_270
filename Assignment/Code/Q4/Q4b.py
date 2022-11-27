# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:38:56 2020
Q4b of E0-270 assignment
@author: Jayabrata
"""

import numpy as np
import dill

filename = "kernel_4a.pkl"
input_file = open(filename,"rb")
kernel_func = dill.loads(dill.load(input_file))

## Defining the distance function  
def dist2(x, Y, m, k):
    mid_term1=0
    for i in range(m):
        mid_term1 += k(x,Y[i])
    mid_term = (2/m)*mid_term1
    last_term1 = 0
    for j in range(m):
        for n in range(m):       # changed from iter count k to n because of call error
            last_term1 += k(Y[j],Y[n]) 
    last_term = (1/(m**2))*last_term1    
    dist2 = (k(x,x) - mid_term + last_term)**0.5 
    return dist2
 
d = 10 # have to change here to 10
I = np.eye(d)
#print(I)

#print(kernel_func(I[0],I[2]))
D =[]
for i in range(d):
    D.append(dist2(I[i], I, d, kernel_func))
#print(D)

## Summing all elements of D 
sum = 0
for j in range(len(D)):
       sum += D[j]
       
print("Final sum of d_i's  in Q4b is: ",sum)
       
    