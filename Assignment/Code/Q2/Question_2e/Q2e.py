# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:48:02 2020
Q2e of E0-270
@author: Jayabrata
"""

import numpy as np
import cvxopt 

# function defition for kernel and optimization
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

# Training data
no_samples = 100
x1x2_min = [-3, -3]
x1x2_max = [3, 3]
x_train = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_samples,2))

label = []
for i in range(no_samples):
    if ((x_train[i][0])**2+((x_train[i][1]**2)*0.5)) <= 2:
        label.append(+1.0)
    else:
        label.append(-1.0)
        
# optimization output       
output = fit(x_train, label, kernel, 0.0)
C = float(input("Enter the hyperparameter C:"))

a_opt=fit(x_train, label, kernel, C)

# finding optimized value of weight  w
w_opt = np.zeros((2, 1)).squeeze()   
for k in range(no_samples):
    w_opt += np.multiply(a_opt[k], np.multiply(x_train[k],label[k]))   # have to multiply a_opt[i]
#print("w_opt is:\n",w_opt)


# finding optimized value of intercept b
neg_max = []
pos_min = []
for l in range(len(label)):
    if label[l] == -1:
        neg_max.append(np.matmul(w_opt.T,x_train[l]))
    else:    
        pos_min.append(np.matmul(w_opt.T,x_train[l]))
       
neg_b = max(neg_max)
pos_b = min(pos_min)
b_opt = -(neg_b + pos_b)*0.5
#print("b_opt is:\n",b_opt)

# Training accuracy
pred1_label=[]
for j in range(no_samples):
    if (np.matmul(w_opt.T,x_train[j] + b_opt) >= 0):
        pred1_label.append(1)
    else:
        pred1_label.append(-1)
wrong1 = 0
for m in range(len(label)):
    if label[m] != pred1_label[m]:
        wrong1 += 1
train_accuracy = ((len(label)-wrong1)/len(label))*100
print("Training accuracy in percentage:  ", train_accuracy)         

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
        
# Testing accuracy
pred2_label=[]
for j in range(no_samples_test):
    if (np.matmul(w_opt.T,x_test[j] + b_opt) >= 0):
        pred2_label.append(1)
    else:
        pred2_label.append(-1)
wrong_test = 0
for m in range(len(label_test)):
    if label_test[m] != pred2_label[m]:
        wrong_test += 1
test_accuracy = ((len(label_test)-wrong_test)/len(label_test))*100
print("Testing accuracy in percentage:  ", test_accuracy)
print("C is :",C)
