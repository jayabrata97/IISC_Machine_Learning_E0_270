# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:46:07 2020

@author: User
"""

import time
import dill
import numpy as np
import matplotlib.pyplot as plt
import sklearn

filename1 = "kernel_4a.pkl"
input_file1 = open(filename1,"rb")
kernel_func = dill.loads(dill.load(input_file1))

filename2 = "data.npy"
data = np.load(filename2)

plt.scatter(data[:,0],data[:,1],marker='x')

## Initializing means
no_means = 2
x1x2_min = [-4, -4]
x1x2_max = [4, 4]
#mu = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_means,2))
#print(mu)
#plt.scatter(mu[:,0],mu[:,1],marker='x')
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)     
y_kmeans = kmeans.predict(data)
plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
