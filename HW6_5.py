import numpy as np
import sklearn
import math as m
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100).T
y_x = np.sin(m.pi*x)

plt.plot(x,y_x)
