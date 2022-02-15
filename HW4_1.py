import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from math import exp

# the logistic function
def logistic_func(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    # split into positive and negative to improve stability
    g[t>=0.0] = 1.0 / (1.0 + np.exp(-t[t>=0.0]))
    g[t<0.0] = np.exp(t[t<0.0]) / (np.exp(t[t<0.0])+1.0)
    return g

# function to compute log-likelihood
def neg_log_like(theta, x, y):
    g = logistic_func(theta,x)
    return -sum(np.log(g[y>0.5])) - sum(np.log(1-g[y<0.5]))

# function to compute the gradient of the negative log-likelihood
#######################################################
#######################################################
##              TODO: PART (C)                       ##
#######################################################
#######################################################
def log_grad(theta, x, y):
    #YOUR IMPLEMENTATION HERE
    return None #RETURN: 1 value: gradient


# implementation of gradient descent for logistic regression
# INPUTS:
#   alpha: step size/learning rate
#   tol: tolerance for GD. If |cost_k - cost_{k-1}| <= tol, STOP.
#           cost_k is the negative log-likelihood with the estimate for \theta at iter k
#   maxiter: maximum number of iterations.
#######################################################
#######################################################
##              TODO: PART (C)                       ##
#######################################################
#######################################################
def grad_desc(theta, x, y, alpha, tol, maxiter):
    #YOUR IMPLEMENTATION HERE
    return None, None #RETURN: 2 values: estimated theta, cost at each iteration as np.array


# function to compute the Hessian of the negative log-likelihood
#######################################################
#######################################################
##              TODO: PART (E)                       ##
#######################################################
#######################################################
def log_hess(theta, x):
    #YOUR IMPLEMENTATION HERE
    return None #RETURN: 1 value: Hessian

# implementation of Newton's method for logistic regression
# INPUTS:
#   tol: tolerance for GD. If |cost_k - cost_{k-1}| <= tol, STOP.
#           cost_k is the negative log-likelihood with the estimate for \theta at iter k
#   maxiter: maximum number of iterations.
#######################################################
#######################################################
##              TODO: PART (E)                       ##
#######################################################
#######################################################
def newton(theta, x, y, tol=1e-12, maxiter=100):
    #YOUR IMPLEMENTATION HERE
    return None, None #RETURN: 2 values: estimated theta, cost at each iteration as np.array

# function to compute output of LR classifier (unused)
def lr_predict(theta,x):
    # form Xtilde for prediction
    shape = x.shape
    Xtilde = np.zeros((shape[0],shape[1]+1))
    Xtilde[:,0] = np.ones(shape[0])
    Xtilde[:,1:] = x
    return logistic_func(theta,Xtilde)

## Generate dataset
np.random.seed(2022) # Set random seed so results are repeatable
x,y = datasets.make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=5.0)

## build classifier
# form Xtilde
shape = x.shape
xtilde = np.zeros((shape[0],shape[1]+1))
xtilde[:,0] = np.ones(shape[0])
xtilde[:,1:] = x

# Initialize theta to zero
theta_gd, theta_newton = np.zeros(shape[1]+1), np.zeros(shape[1]+1)

# Run gradient descent
alpha = 1e-5
maxiter = 10000
tol = 1e-3

import time
start = time.time()
theta_gd, cost_gd = grad_desc(theta_gd,xtilde,y,alpha,tol,maxiter)
end = time.time()
print('Running time of GD: ' + str(end-start))
print('Final value of negative log-likelihood for GD: ' + str(cost_gd[-1]))
print('Number of iterations for GD: ' + str(len(cost_gd)-1))

#UNCOMMENT TO RUN NEWTON
#start = time.time()
#theta_newton, cost_newton = newton(theta_newton, xtilde, y, tol=tol, maxiter=maxiter)
#end = time.time()
#print('Running time of Newton: ' + str(end-start))
#print('Final value of negative log-likelihood of Newton: ' + str(cost_newton[-1]))
#print('Number of iterations for Newton: ' + str(len(cost_newton)-1))


plt.plot(np.arange(len(cost_gd)), cost_gd)
#plt.plot(np.arange(len(cost_newton)), cost_newton)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Logistic regression cost vs. iterations")
plt.show()
