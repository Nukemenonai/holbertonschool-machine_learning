#!/usr/bin/env python3
"""
Expectation
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates the expectation step in the EM algorithm for a GMM:

    X:(n, d) containing the data set
    pi: (k,) containing the priors for each cluster
    m: (k, d) containing the centroid means for each cluster
    S:(k, d, d) containing the covariance matrices for each cluster
    Returns: g, l, or None, None on failure
    g: (k, n) the posterior probabilities for each data point in each cluster
    l: total log likelihood
    """

    if not type(X) == np.ndarray or len(X.shape) != 2:
        return None, None
    if not type(m) == np.ndarray or len(m.shape) != 2:
        return None, None
    if not type(S) == np.ndarray or len(S.shape) != 3:
        return None, None
    if not type(pi) == np.ndarray or len(pi.shape) != 1:
        return None, None

    n, _ = X.shape
    k, d, _ = S.shape

    g = np.zeros((k, n))

    for cluster in range(k):
        prob = pdf(X, m[cluster], S[cluster])
        prior = pi[cluster]
        g[cluster] = prior * prob

    total = np.sum(g, axis=0, keepdims=True)

    posterior = g / total

    gmm_prob = np.sum(np.log(total))

    return posterior, gmm_prob
