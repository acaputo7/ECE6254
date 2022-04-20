from sklearn.datasets import load_digits
from sklearn import manifold, decomposition
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from itertools import zip_longest

#Input:     X - learned embedding, size n_samples x 2
#           title - title of embedding visualization.
# def plot_embedding(X, title):
#     fig, ax = plt.subplots()
#
#     X = MinMaxScaler().fit_transform(X)
#     for i in range(X.shape[0]):
#         imagebox = offsetbox.AnnotationBbox(
#             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#             X[i],
#             bboxprops = dict(color=plt.cm.Dark2(y[i]))
#         )
#         imagebox.set(zorder=1)
#         ax.add_artist(imagebox)
#
#     plt.title(title)
#     plt.axis("off")
#     #plt.savefig(title)
#     plt.show()
def plot_embedding(X, title, ax):
    X = MinMaxScaler().fit_transform(X)
    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")

np.random.seed(2022)

digits = load_digits(n_class=7)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n_neighbors = 30
embeddings = {
    "PCA embedding": decomposition.PCA(n_components=2
    ),
    "IL Kernel PCA embedding": decomposition.KernelPCA(n_components=2, kernel='poly', degree=1
    ),
    "Quadratic Kernel PCA embedding": decomposition.KernelPCA(n_components=2, kernel='poly', degree=2
    ),
    "MDS embedding": manifold.MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2
    ),

    "LLE embedding": manifold.LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="standard"
    ),
    "Isomap embedding": manifold.Isomap(n_neighbors=n_neighbors, n_components=2
    ),
}

projections, timing = {}, {}
for name, transformer in embeddings.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    print(f"Computing {name}...")
    start_time = time()
    projections[name] = transformer.fit_transform(data, y)
    timing[name] = time() - start_time



fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 16))

for name, ax in zip_longest(timing, axs.ravel()):
    if name is None:
        ax.axis("off")
        continue
    title = f"{name} (time {timing[name]:.3f}s)"
    plot_embedding(projections[name], title, ax)

plt.show()
## PCA


## Kernel PCA


## MDS


## LLE


## Isomap
