# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:45:29 2020
Q3a of E0-270 assignment
@author: Jayabrata
"""
import numpy as np
import matplotlib.pyplot as plt

no_samples = 100
x_min = [-1]
x_max = [1]
x = np.random.uniform(low = x_min, high = x_max, size = (no_samples,1))
#print(x)
y = []
for i in range(no_samples):
    y.append(np.sin(3*x[i]).tolist())
#print(y)
plt.scatter(x, y, marker='x')
plt.xlabel("x_i")
plt.ylabel("y_i")
plt.grid()
plt.title("Plot of Q3a")
plt.show()
    