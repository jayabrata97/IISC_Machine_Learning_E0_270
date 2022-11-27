# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:56:45 2020
Q5a of E0-270 assignment
@author: Jayabrata
"""
import numpy as np
import matplotlib.pyplot as plt

no_samples = 100 # have to change here to 100
x11x12_min = [-1, 0]
x11x12_max = [1, 0]
C1 = np.random.uniform(low = x11x12_min, high = x11x12_max, size = (no_samples,2))
#print(C1) 
no_samples2 = round(np.random.uniform(low = 0, high = 100))
print("no_samples2: ",no_samples2)

x2x2_min = [-3, 0]
x2x2_max = [-2, 0]  
C21 = np.random.uniform(low = x2x2_min, high = x2x2_max, size = (no_samples2,2))
#print(len(C21))
x21x22_min = [2, 0]
x21x22_max = [3, 0]  
C22 = np.random.uniform(low = x21x22_min, high = x21x22_max, size = ((100-no_samples2),2)) 
#print(C22)
C2 = np.concatenate((C21,C22))

## Plotting data
plt.scatter(C1[:,0], C1[:,1], c='red', marker ='x')
plt.scatter(C2[:,0], C2[:,1], c='blue', marker = 'x')
plt.title("Plot of Q5a, C1: red, C2: blue")
plt.grid(True)
plt.show()
 





