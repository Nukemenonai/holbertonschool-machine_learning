#!/usr/bin/env python3
"""
EM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM:

    X: (n, d) the data set
    k: positive integer, the number of clusters
    iterations: positive integer,the maximum number
    of iterations for the algorithm
    tol is a non-negative float containing tolerance 
    of the log likelihood, used to determine early stopping 
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    # L = likelihood
    prev_L = 0

    for i in range(iterations):
        g, L = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)

        if verbose:
            message = 'Log Likelihood after {} iterations: {}'\
                .format(i, L.round(5))
            if (i % 10 == 0) or (i == 0):
                print(message)

            if abs(L - prev_L) <= tol:
                print(message)
                break
        if abs(L - prev_L) <= tol:
            break
        prev_L = L
    return pi, m, S, g, L
