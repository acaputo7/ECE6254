from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math

def k_x(x, Xtrain, gamma):
    k = np.zeros(len(Xtrain))
    for i in range(len(Xtrain)):
        k[i] = np.exp(-gamma * LA.norm((Xtrain[i] - x)) ** 2)
        kx = k.T
    return kx

def krr_predict(Xtrain, ytrain, Xtest, gamma, lamb):
    Lamb = math.sqrt(lamb) * np.identity(len(Xtrain))
    Ktrain = np.zeros((len(Xtrain),len(Xtrain)))
    for i in range(len(Xtrain)):
        for j in range(len(Xtrain)):
            Ktrain[i,j] = np.exp(-gamma * LA.norm((Xtrain[i] - Xtrain[j])) ** 2)
    kx = k_x(Xtrain, Xtest, gamma)
    fx = np.zeros(len(Xtest))
    for f,x in enumerate(Xtest):
        kx = k_x(x,Xtrain,gamma)
        fx[f] = ytrain.T @ LA.inv(Ktrain + Lamb) @ kx
    return fx

#establish the training data
np.random.seed(2022)
n = 100
Xtrain = np.random.rand(n)
ytrain = np.sin(9*Xtrain) + np.sqrt(1/3.0)*np.random.randn(n)


#create test data to quantify quality of fit
Xtest = np.linspace(0,1,1001)
ytest = np.sin(9*Xtest)

##  3a  ##
gamma = 5
lamb = 0.01

ypred = krr_predict(Xtrain, ytrain, Xtest, gamma, lamb)

plt.figure(1)
plt.scatter(Xtrain, ytrain, label='Train')
plt.plot(Xtest, ytest, label='Test')
plt.scatter(Xtest, ypred, label='KRR')
plt.legend()

plt.show()
