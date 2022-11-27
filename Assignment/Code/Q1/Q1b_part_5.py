# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:34:03 2020
Qno: 1b part v
@author: Jayabrata
"""
import dill
import numpy as np


filename1 = "function5.pkl" 
input_file1 = open(filename1,"rb")
func1 = dill.loads(dill.load(input_file1))


filename2 = "k5sampler.pkl" 
input_file2 = open(filename2,"rb")
k5sampler = dill.loads(dill.load(input_file2))

sampler = k5sampler()

# Taking samples here
no_samples = 100
X=[]
K_mat=[]
for k in range(no_samples):
    X.append(sampler)
#print(X)    
for i in range(no_samples):
    for j in range(no_samples):
        K_mat.append(func1(X[i],X[j]))
        
K_mat = np.reshape(K_mat,(no_samples,-1))          
#print(K_mat) 
label =[]      
for m in range(len(X)):
    for n in range(len(X)):
        if func1(X[m],X[n]) == func1(X[n],X[m]) and func1(X[m],X[m]) >= 0:
            label.append(1)
        else:
            label.append(-1)
#print(label)
for p in range(no_samples**2):
    if label[p] == -1:
        print('it is not a kernel')    
        break 
    
def is_psd(X):
    return np.all(np.linalg.eigvals(X)>=0.0)
print(is_psd(K_mat))
if is_psd(K_mat) == True:
    print('k5 is a valid kernel')
else:
    print("k5 is not valid kernel")
    



