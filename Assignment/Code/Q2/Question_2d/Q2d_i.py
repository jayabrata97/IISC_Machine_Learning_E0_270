# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:12:03 2020
Q2d part i of E0-270 assignment
@author: Jayabrata
"""
import numpy as np
import matplotlib.pyplot as plt

no_samples = 100 # have to change here to 100
x1x2_min = [-3, -3]
x1x2_max = [3, 3]
x = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_samples,2))
#print(x)

label = []
for i in range(no_samples):
    if ((x[i][0])**2+((x[i][1]**2)*0.5)) <= 2:
        label.append(+1.0)
    else:
        label.append(-1.0)
#print(x[0][1])        
phi_x = []
for i in range(no_samples):
    for j in range(2):
        phi_x.append((x[i][j])**2)
phi_x = np.reshape(phi_x,(no_samples,-1))
#print(phi_x)        
        
# Preparing data for plotting 

positive = []
negative = []
for j in range(len(label)):
    if label[j] == 1:
        positive.append(phi_x[j].tolist())
    else:
        negative.append(phi_x[j].tolist()) 
positive_x1=[]
negative_x1=[]
for k in range(x.shape[0]):
    if np.any(np.all(np.isin(positive,phi_x[k],True),axis=1)):
        positive_x1.append(phi_x[k][0])
    else:
        negative_x1.append(phi_x[k][0])                
positive_x2=[]
negative_x2=[]
for l in range(x.shape[0]):
    if np.any(np.all(np.isin(positive,phi_x[l],True),axis=1)):
        positive_x2.append(phi_x[l][1])
    else:
        negative_x2.append(phi_x[l][1])  
        
# Plot the scatter plot       
plt.scatter(positive_x1,positive_x2,c='red',marker='x')
plt.scatter(negative_x1,negative_x2,c='blue',marker='x')
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("Scatter plot for Q2d_part1, Red = +1.0, Blue = -1.0")
plt.show()         
        
