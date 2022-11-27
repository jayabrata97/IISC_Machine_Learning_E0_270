# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:24:22 2020

@author: User
"""
import numpy as np
#def feature_map(k, x, no_samples):
def phi_tilda(x, no_samples,fea_dim):
    phi = []
    for i in range(no_samples):
        for j in range(fea_dim): 
            phi.append((x[i]**(j+1)))       
    phi = np.reshape(phi,(no_samples,-1))        
    x0 = np.ones((no_samples, 1))
    phi_tilda = np.hstack((x0,phi))    
    
    return phi_tilda

no_samples = 3  # it should be changed to 100
x_min = [-1]
x_max = [1]
x = np.random.uniform(low = x_min, high = x_max, size = (no_samples,1))

fea_dim = 3   # have to change to 10 here 
for k in range(fea_dim):
    phi_tilda = phi_tilda(x, no_samples, (k+1))
    print('phi_tilda:\n',phi_tilda)