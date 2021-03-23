#!/usr/bin/env python3
""" this module finds the mean and covariance of a data set"""

import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    
    mean = np.expand_dims(np.mean(X, axis=0), axis=0)
    X -= mean

    cov = np.dot(X.T, X) / (n - 1)

    return mean, cov
