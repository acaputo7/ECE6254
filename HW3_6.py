from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
from sklearn import linear_model

california = fetch_california_housing()

# Import the data and assign variables
    # X is a 20640 x 8 matrix
X = np.array(california.data)
Y = np.array(california.target)

# Split the data into test and train pairs of X and Y with there being 1000 training points
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=1000/len(X), random_state=42)

# Train Scaler on training data
scaler = preprocessing.StandardScaler().fit(X_train)

# Execute trained scaler on training and testing X data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

##6a) OLS
# Figure out theta_hat using X (scaled) and Y training data using OLS
theta_hat_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train_scaled.T, X_train_scaled)), X_train_scaled.T), y_train)

# Compute MSE of trained theta_hat when applied to test data
MSE_OLS = (1/len(X_test_scaled))*np.linalg.norm(y_test-np.matmul(X_test_scaled, theta_hat_OLS))**2
print(MSE_OLS)

#6b) Ridge regression
# Figure out gamma value for theta_hat using X (scaled) and Y training data using Ridge Regression by picking one with
# lowest MSE on CV, but also validating it has a minimum MSE for test data

# for gamma in range(1,100):
#     Gamma = math.sqrt(gamma) * np.identity(8)
#     Gamma[0, 0] = 0
#     X_trainCV, X_testCV, y_trainCV, y_testCV = train_test_split(X_train_scaled, y_train, test_size=10, random_state=42)
#
#
#     theta_hat_ridgeCV = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_trainCV.T, X_trainCV)+np.matmul(Gamma.T, Gamma)), X_trainCV.T), y_trainCV)
#
#
#     MSE_ridge_train = (1/len(X_testCV))*np.linalg.norm(y_testCV-np.matmul(X_testCV, theta_hat_ridgeCV))**2
#     print('for gamma of ' + str(gamma) + ' \nMSE = ' + str(MSE_ridge_train))

gamma = 34
Gamma = math.sqrt(gamma) * np.identity(8)
Gamma[0, 0] = 0

# Use gamma value from CV to calculate theta_hat using X (scaled) and Y training data using Ridge Regression
theta_hat_ridge = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train_scaled.T, X_train_scaled)+np.matmul(Gamma.T, Gamma)), X_train_scaled.T), y_train)

#Compute MSE of trained theta_hat when applied to test data
MSE_ridge = (1/len(X_test_scaled))*np.linalg.norm(y_test-np.matmul(X_test_scaled, theta_hat_ridge))**2
print(MSE_ridge)

#6c) LASSO

# Train model using self encoded CV to see what the MSE of a good alpha value is
reg2 = linear_model.LassoCV(cv=10, random_state=0).fit(X_train_scaled, y_train)
y_pred = reg2.predict(X_test_scaled)
MSE_lasso_train2 = (1/len(X_test_scaled))*np.linalg.norm(y_test-y_pred)**2
print('for alpha of ' + str(reg2.get_params()) + ' \nMSE = ' + str(MSE_lasso_train2))
print(reg2.coef_)

# Plug in values of alpha until MSE matches that of the LassoCV model
alpha = 0.0088
reg = linear_model.Lasso(alpha=alpha)
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
MSE_lasso_train = (1 / len(X_test_scaled)) * np.linalg.norm(y_test - y_pred) ** 2
print('for alpha of ' + str(alpha) + ' \nMSE = ' + str(MSE_lasso_train))
print(reg.coef_)
