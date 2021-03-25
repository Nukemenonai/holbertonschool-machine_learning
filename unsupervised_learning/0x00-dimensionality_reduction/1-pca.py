#!/usr/bin/env python3
""" PCA with X transformation """

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    X_mean = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(X_mean)

    W = Vh.T
    Wr = W[:, :ndim]

    Tr = np.dot(X_mean, Wr)
    return Tr
