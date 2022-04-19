from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2022)

## PART A: CLASSIFICATION
digits = load_digits(n_class=10)
X, y = digits.data, digits.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Fit MLPClassifier to the training data
clf = MLPClassifier(max_iter=1000, activation='relu')
clf.fit(X_train, y_train)

# Use the trained MLPClassifier to predict the class labels
y_pred = clf.predict(X_test)

# Compute the accuracy on the test set
score = clf.score(X_test, y_test)
print('The final loss of the trained classifier is: ' + '{:.5f}'.format(clf.loss_))
print('number of layers= ' + str(clf.n_layers_) + ' with an accuracy of ' + str('{:.3f}'.format(score)))
print(clf.get_params())

## PART B: REGRESSION
n = 300
xtrain = np.sort(np.random.rand(n))
ytrain = np.sin(9*xtrain) + xtrain + np.sqrt(1/5.0)*np.random.randn(n)
xtest = np.linspace(0,1,1001)
ytest = np.sin(9*xtest) + xtest

# Fit MLPRegressor to training data.
reg = MLPRegressor(activation='relu', max_iter=1000, hidden_layer_sizes=100)
reg.fit(xtrain.reshape(-1,1), ytrain)
print(reg.n_layers_)

# Use the trained MLPRegressor to predict regression outputs
ypred = reg.predict(xtest.reshape(-1,1))
print(reg.get_params())
# Compute mean squared error on test set, plot network output.
MSE = mean_squared_error(y_true=ytest,y_pred=ypred)
print('The MSE is: ' + str('{:.4f}'.format(MSE)))


plt.figure(1)
plt.plot(xtest, ytest)
plt.plot(xtest, ypred)
plt.show()