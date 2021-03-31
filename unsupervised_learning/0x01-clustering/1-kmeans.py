#!/usr/bin/env python3
""" K-means """

import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means
    X: numpy.ndarray of shape (n, d) with the dataset that will be used
    for K-means
    k: positive integer containing the number of clusters
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(k) is not int or k <= 0:
        return None

    d = X.shape[1]

    # random centroid initialization
    # with min and max values of X per column

    minX = np.min(X, axis=0).astype(np.float)
    maxX = np.max(X, axis=0).astype(np.float)
    centroids = np.random.uniform(low=minX, high=maxX, size=(k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    centroids = initialize(X, k)
    clss = None

    for _ in range(iterations):
        D = np.linalg.norm(X[:, None] - centroids, axis=-1)
        clss = np.argmin(D, axis=-1)
        c_copy = np.copy(centroids)
        for j in range(k):
            idx = np.argwhere(clss == j)
            if not len(idx):
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = np.mean(X[idx], axis=0)
        if (c_copy == centroids).all():
            return centroids, clss
    D = np.linalg.norm(X[:, None] - centroids, axis=-1)
    clss = np.argmin(D, axis=-1)
    return centroids, clss
