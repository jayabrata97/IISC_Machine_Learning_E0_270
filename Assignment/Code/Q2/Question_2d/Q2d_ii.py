# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:54:49 2020
Q2d part ii of E0-270 assignment
@author: Jayabrata
"""

import numpy as np
import cvxopt

## Training data
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

def kernel(x1,x2):
    return np.dot(x1,x2)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

def fit(X, y, polynomial_kernel, C):
    n_samples,n_features=X.shape 
    # Gram matrix
    K=np.zeros((n_samples,n_samples))  
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j]=kernel(X[i],X[j])
    # parameters of optimization
    P=cvxopt.matrix(np.outer(y,y)*K)
    q=cvxopt.matrix(np.ones(n_samples)*-1)
    A=cvxopt.matrix(y,(1,n_samples))
    b=cvxopt.matrix(0.0)
    if C is None:
        G=cvxopt.matrix(np.diag(np.ones(n_samples)*-1))
        h=cvxopt.matrix(np.zeros(n_samples))
    else:
        G=cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples)*-1),np.identity(n_samples))))
        h=cvxopt.matrix(np.hstack((np.zeros(n_samples),np.ones(n_samples)*C)))
    # solve QP 
    solution=cvxopt.solvers.qp(P,q,G,h,A,b)
    # Lagrange multipliers
    a=np.ravel(solution['x'])  
    return a

output = fit(phi_x, label, kernel, 0.0)
C = float(input("Enter the hyperparameter C:"))
a_opt=fit(phi_x, label, kernel, C)
w_opt = np.zeros((2, 1)).squeeze()   
for k in range(no_samples):
    w_opt += np.multiply(a_opt[k], np.multiply(phi_x[k],label[k]))   # have to multiply a_opt[i]
print("\nw_opt is:\n",w_opt)
neg_max = []
pos_min = []
for l in range(len(label)):
    if label[l] == -1:
        neg_max.append(np.matmul(w_opt.T,phi_x[l]))
    else:    
        pos_min.append(np.matmul(w_opt.T,phi_x[l]))
neg_b = max(neg_max)
pos_b = min(pos_min)
b_opt = -(neg_b + pos_b)*0.5
print("\nb_opt is:\n",b_opt)   


# Training accuracy
pred1_label=[]
for j in range(no_samples):
    if (np.matmul(w_opt.T,phi_x[j] + b_opt) >= 0):
        pred1_label.append(1)
    else:
        pred1_label.append(-1)
wrong1 = 0
for m in range(len(label)):
    if label[m] != pred1_label[m]:
        wrong1 += 1 
train_accuracy = ((len(label)-wrong1)/len(label))*100
print("\nTraining accuracy in percentage:  ", train_accuracy) 

# Testing data
no_samples_test = 50
x1x2_min = [-3, -3]
x1x2_max = [3, 3]
x_test = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_samples,2))
        
label_test = []
for i in range(no_samples_test):
    if ((x_test[i][0])**2+((x_test[i][1]**2)*0.5)) <= 2:
        label_test.append(+1.0)
    else:
        label_test.append(-1.0)
phi_x_test = []
for i in range(no_samples_test):
    for j in range(2):
        phi_x_test.append((x_test[i][j])**2)
phi_x_test = np.reshape(phi_x_test,(no_samples_test,-1))   


pred2_label=[]
for j in range(no_samples_test):
    if (np.matmul(w_opt.T,phi_x_test[j] + b_opt) >= 0):
        pred2_label.append(1)
    else:
        pred2_label.append(-1) 
wrong_test = 0 
for m in range(len(label_test)):
    if label_test[m] != pred2_label[m]:
        wrong_test += 1
test_accuracy = ((len(label_test)-wrong_test)/len(label_test))*100
print("\nTesting accuracy in percentage:  ", test_accuracy)     
print("\nHyperparameter C is: ",C)


