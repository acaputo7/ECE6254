from sklearn.datasets import load_digits
from sklearn import manifold, decomposition

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler

#Input:     X - learned embedding, size n_samples x 2
#           title - title of embedding visualization.
def plot_embedding(X, title):
    fig, ax = plt.subplots()

    X = MinMaxScaler().fit_transform(X)
    for i in range(X.shape[0]):
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            X[i],
            bboxprops = dict(color=plt.cm.Dark2(y[i]))
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    plt.title(title)
    plt.axis("off")
    #plt.savefig(title)
    plt.show()


np.random.seed(2022)

digits = load_digits(n_class=7)
X, y = digits.data, digits.target
n_samples, n_features = X.shape

## PCA


## Kernel PCA


## MDS


## LLE


## Isomap
