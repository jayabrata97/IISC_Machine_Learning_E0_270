# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:12:09 2020
Q2b of E0-270 Assignment
@author: Jayabrata
"""

import numpy as np
import cvxopt 


# Preparing training data
mean = [0,0]
cov = np.eye(2)
 
#Sampling w form normal distribution
w1,w2 = np.random.multivariate_normal(mean,cov,1).T
w_train = np.array([w1,w2]).squeeze()


# Sampling b from normal distribution
b_train = np.random.normal(0,1,size=1)

# Sampling data point x from uniform distribution
no_samples = 100 
x1x2_min = [-3, -3]
x1x2_max = [3, 3]
x_train = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_samples,2))

# Calculating sign for samples of data
scores = []
label = []
for i in range(no_samples):
    scores.append(np.matmul(w_train.T, x_train[i]) + b_train)
    if scores[i] > 0:
        label.append(+1.0)
    else:
        label.append(-1.0)

def kernel(x1,x2):
    return np.dot(x1,x2)

def fit(X, y ,kernel, C):
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
# optimization output       
output = fit(x_train, label, kernel, 0.0)
C = float(input("Enter hyper-parameter C: ")) # for 100 i got good accuracy
a_opt=fit(x_train, label, kernel, C)


# finding optimized value of weight  w
w_opt = np.zeros((2, 1)).squeeze()   
for k in range(no_samples):
    w_opt += np.multiply(a_opt[k], np.multiply(x_train[k],label[k]))   # have to multiply a_opt[i]
print("\nw_opt is:\n",w_opt)


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
print("\nb_opt is:\n",b_opt)


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
print("\nTraining accuracy in percentage:  ", train_accuracy)         


## Testing data 
#Sampling w form normal distribution
w1,w2 = np.random.multivariate_normal(mean,cov,1).T
w_test = np.array([w1,w2]).squeeze()


# Sampling b from normal distribution
b_test = np.random.normal(0,1,size=1)

# Sampling data point x from uniform distribution
no_samples_test = 50     
x1x2_min = [-3, -3]
x1x2_max = [3, 3]
x_test = np.random.uniform(low = x1x2_min, high = x1x2_max, size = (no_samples,2))

# Calculating sign for samples of data
scores_test = []
label_test = []
for i in range(no_samples_test):
    scores_test.append(np.matmul(w_test.T, x_test[i]) + b_test)
    if scores_test[i] > 0:
        label_test.append(+1.0)
    else:
        label_test.append(-1.0)

# Testing accuracy
pred2_label=[]
for n in range(no_samples_test):
    if (np.matmul(w_opt.T,x_test[n] + b_opt) >= 0):
        pred2_label.append(1)
    else:
        pred2_label.append(-1)
wrong_test = 0
for p in range(len(label_test)):
    if label_test[p] != pred2_label[p]:
        wrong_test += 1
test_accuracy = ((len(label_test)-wrong_test)/len(label_test))*100
print("\nTesting accuracy in percentage:  ", test_accuracy)
print("\nHyperparameter C is: ",C)

 

        






    