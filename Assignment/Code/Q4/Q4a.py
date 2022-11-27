# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:01:26 2020
Q4a of E0-270 Assignment 
@author: Jayabrata
"""
import dill
import numpy as np

filename = "kernel_4a.pkl"
input_file = open(filename,"rb")
kernel_func = dill.loads(dill.load(input_file))

## Defining distance function
def dist(x, y , k):
    dist = round((k(y,y) - 2*k(x,y) + k(x,x))**0.5)
    return dist

## dataset with e
d = 10 # have to change here to 10
I = np.eye(d)
#print(I)
e = []
for i in range(d):
    e.append(I[i])
e = np.reshape(e,(d,-1))
#print(e)    

## Distance matrix
D =[]
for j in range(d):
    for k in range(d):
        D.append(dist(e[j], e[k], kernel_func))
D = np.reshape(D,(d,d))        
#print(D)
sum=0
for l in range(d):
    for m in range(d):
        sum += D[l][m]
print("sum of all elements of D is: ",sum)    
   

         