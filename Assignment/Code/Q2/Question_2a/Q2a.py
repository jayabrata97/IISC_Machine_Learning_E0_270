# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:40:36 2020
Completed Qno 2a of assignment E0-270
@author: Jayabrata
"""

import numpy as np
import matplotlib.pyplot as plt

# Prameters of 2D normal distribution of w
mean = [0,0]
cov = np.eye(2)
#print(cov)

#Sampling w form normal distribution
w1,w2 = np.random.multivariate_normal(mean,cov,1).T
w = np.array([w1,w2]).squeeze()
#plt.plot(w1,w2,'x')
#plt.axis('equal')
#plt.xlabel('w1')
#plt.ylabel('w2')
#plt.show() 

# Sampling b from normal distribution
b = np.random.normal(0,1,size=1)


# Sampling data point x from uniform distribution
no_samples = 100
x1x2_min = [-3, -3]
x1x2_max = [3, 3]
x = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_samples,2))
#print(x)
#plt.plot(x)

# Calculating sign for samples of data
scores = []
label = []
for i in range(no_samples):
    scores.append(np.matmul(w.T, x[i]) + b)
    if scores[i] > 0:
        label.append(+1.0)
    else:
        label.append(-1.0)
 
# Preparing data for plotting 

positive = []
negative = []
for j in range(len(label)):
    if label[j] == 1:
        positive.append(x[j].tolist())
    else:
        negative.append(x[j].tolist()) 
positive_x1=[]
negative_x1=[]
for k in range(x.shape[0]):
    if np.any(np.all(np.isin(positive,x[k],True),axis=1)):
        positive_x1.append(x[k][0])
    else:
        negative_x1.append(x[k][0])                
positive_x2=[]
negative_x2=[]
for l in range(x.shape[0]):
    if np.any(np.all(np.isin(positive,x[l],True),axis=1)):
        positive_x2.append(x[l][1])
    else:
        negative_x2.append(x[l][1])  
        
# Plot the scatter plot       
plt.scatter(positive_x1,positive_x2,c='red',marker='x')
plt.scatter(negative_x1,negative_x2,c='blue',marker='x')
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("Scatter plot for Q2a, Red = +1.0, Blue = -1.0")
plt.show()   