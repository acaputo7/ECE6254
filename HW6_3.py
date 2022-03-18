from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
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


#3b
e = 0.1
g = 10
Cvec = np.logspace(0,3,100)
Cmae = np.ones(len(Cvec))
for i in range(len(Cvec)) :
    reg = SVR(C=Cvec[i], epsilon=e, kernel='rbf', gamma=g)
    reg.fit(Xtrain.reshape(-1,1), ytrain)
    ypred = reg.predict(Xtest.reshape(-1,1))
    mae = mean_absolute_error(ytest,ypred)

    Cmae[i] = mean_absolute_error(ytest,ypred)
    # print('For Cval = ' + str(Cvec[i]) + ' MAE = ' + str(mae))
Copt = Cvec[np.argmin(Cmae)]
print('The optimal C = ' + str(Copt) + ' with MAE = ' + str(np.min(Cmae)))

evec = np.logspace(-2,3,100)
emae = np.ones(len(evec))
for i in range(len(evec)) :
    reg = SVR(C=Copt, epsilon=evec[i], kernel='rbf', gamma=g)
    reg.fit(Xtrain.reshape(-1,1), ytrain)
    ypred = reg.predict(Xtest.reshape(-1,1))
    mae = mean_absolute_error(ytest,ypred)

    emae[i] = mean_absolute_error(ytest,ypred)
    # print('For eval = ' + str(evec[i]) + ' MAE = ' + str(mae))
eopt = evec[np.argmin(emae)]
print('The optimal e = ' + str(eopt) + ' with MAE = ' + str(np.min(emae)))

gvec = np.logspace(-2,3,100)
gmae = np.ones(len(gvec))
for i in range(len(gvec)) :
    reg = SVR(C=Copt, epsilon=eopt, kernel='rbf', gamma=gvec[i])
    reg.fit(Xtrain.reshape(-1,1), ytrain)
    ypred = reg.predict(Xtest.reshape(-1,1))
    mae = mean_absolute_error(ytest,ypred)

    gmae[i] = mean_absolute_error(ytest,ypred)
    # print('For gval = ' + str(gvec[i]) + ' MAE = ' + str(mae))
gopt = gvec[np.argmin(gmae)]
print('The optimal g = ' + str(gopt) + ' with MAE = ' + str(np.min(gmae)))


reg = SVR(C=Copt, epsilon=eopt, kernel='rbf', gamma=gopt)
reg.fit(Xtrain.reshape(-1,1), ytrain)
ypred = reg.predict(Xtest.reshape(-1,1))

plt.figure(2)
plt.scatter(Xtrain, ytrain, label='Train')
plt.plot(Xtest, ytest, label='Test')
plt.scatter(Xtest, ypred, label='SVR with RBF')
plt.legend()
plt.show()