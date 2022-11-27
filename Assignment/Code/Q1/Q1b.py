# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:39:02 2020
Q1b of E0-270 assignment
@author: Jayabrata
"""

import dill
import numpy as np

filename = "function1.pkl"  # change filename to function*.pkl for each file
input_file = open(filename,"rb")
func = dill.loads(dill.load(input_file))

n = 400 # have to change here to 100
X=[]
np.random.seed(100)
for i in range(0,n):
    X.append(np.random.uniform(low=-5,high=5, size=(3,1)))
    
K_mat = np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        K_mat[i,j]=func(X[i],X[j])
        
is_sym = (K_mat.transpose() == K_mat).all()

eig_val,_ = np.linalg.eig(K_mat)
is_psd = ((eig_val) >= -0.0001).all()

if is_sym and is_psd:
    print("It is valid kernel")
else:
    print("It is not a valid kernel")
    
         
    
      







