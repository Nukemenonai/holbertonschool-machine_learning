#!/usr/bin/env python3
"""optimize k"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum numbers of clusters by variance
    X: dataset
    kmin: minimum number of clusters to check for
    kmax: maximum number of clusters to check for
    iterations: maximum number of iterations for k-means
    Return: results, d_vars. None, None on failure.
        results: outputs of K-means for each cluster size
        d_vars: difference in variance from the smallest cluster size
        for each cluster size
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or type(iterations) is not int:
        return None, None
    if kmax is not None and type(kmax) is not int:
        return None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if kmin <= 0 or kmax <= 0 or iterations <= 0 or kmin >= kmax:
        return None, None

    results, d_vars = [], []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var = variance(X, C)
        if k == kmin:
            min_var = var
        d_vars.append(min_var - var)
    return results, d_vars
