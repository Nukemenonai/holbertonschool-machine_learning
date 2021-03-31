#!/usr/bin/env python3
""" initialize k-means"""

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
