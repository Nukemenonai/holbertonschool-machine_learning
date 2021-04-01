#!/usr/bin/env python3
"""
maximization
"""

import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM:

    X: (n, d) the data set
    g: (k, n) the posterior probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
        pi: (k,) the updated priors for each cluster
        m: (k, d) the updated centroid means for each cluster
        S: (k, d, d) the updated covariance matrices for each cluster
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None

    if X.shape[0] != g.shape[1]:
        return None, None, None

    # sum per cluster == 1 => sum of all is n
    summ = np.sum(g, axis=0)
    summ = np.sum(summ)
    if (int(summ) != X.shape[0]):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    N_soft = np.sum(g, axis=1)
    pi = N_soft / n

    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for cluster in range(k):
        rik = g[cluster]
        den = N_soft[cluster]
        mean[cluster] = np.matmul(rik, X) / den
        # cov
        # we have to use element wise first to keep (d, n) by broadcasting
        # then we can use the matrix multiplication to get (d, d) dims
        first = rik * (X - mean[cluster]).T
        cov[cluster] = np.matmul(first, (X - mean[cluster])) / den

    return (pi, mean, cov)
    