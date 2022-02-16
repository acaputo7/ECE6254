import numpy as np
import sklearn
from sklearn import linear_model
import math as m
import matplotlib.pyplot as plt

X = np.linspace(-1,1,100)
y_x = np.sin(m.pi*X)
print(X)
# reg = linear_model.LinearRegression().fit(X, y_x)

# y_pred = reg.predict([X])


plt.plot(X, y_x)
# plt.plot(X, y_pred)
plt.show()


