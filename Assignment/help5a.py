# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:55:01 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

no_samples = 10 # have to change here to 100
x11x12_min = [-1, 0]
x11x12_max = [1, 0]
C1 = np.random.uniform(low = x11x12_min, high = x11x12_max, size = (no_samples,2))
#print(C1)
#no_samples2=5
#x2x2_min = [-3, 0]
#x2x2_max = [-2, 0]  
#C21 = np.random.uniform(low = x2x2_min, high = x2x2_max, size = (no_samples2,2))
#print(C21)
#x21x22_min = [2, 0]
#x21x22_max = [3, 0]  
#C22 = np.random.uniform(low = x21x22_min, high = x21x22_max, size = (no_samples2,2)) 
#print(C22)
#C2 = np.concatenate((C21,C22)) 
#print(C2)

x21x22_min = [[-3,0],[2,0]]
x21x22_max = [[-2,0],[3,0]]
choice_indices_low = np.random.choice(len(x21x22_min))
low = x21x22_min[choice_indices_low]
print(low)
choice_indices_high = np.random.choice(len(x21x22_max))
high = x21x22_max[choice_indices_high]
print(high)

C2 = np.random.uniform(low = low, high = high, size = (no_samples,2))
#print(C2)

plt.scatter(C1[:,0], C1[:,1], c='red', marker ='x')
#plt.scatter(C21[:,0], C21[:,1], c='blue', marker = 'x')
#plt.scatter(C22[:,0], C22[:,1], c='blue', marker = 'x')
plt.scatter(C2[:,0], C2[:,1], c='blue', marker = 'x')
plt.title("Plot of Q5a, C1: red, C2: blue")
plt.grid(True)
plt.show()

