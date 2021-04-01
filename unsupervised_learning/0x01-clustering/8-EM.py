#!/usr/bin/env python3
"""
Expectation Maximization 
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
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None, None)

    if type(k) is not int or k <= 0:
        return (None, None, None, None, None)

    if type(iterations)is not int or iterations <= 0:
        return (None, None, None, None, None)

    if type(tol) is not float or tol < 0:
        return (None, None, None, None, None)

    if type(verbose) is not bool:
        return (None, None, None, None, None)

    pi, m, S = initialize(X, k)
    prev_L = 0
    i = 0

    g, L = expectation(X, pi, m, S)
    while(i < iterations):
        if (np.abs(prev_L - L)) <= tol:
            break

        if verbose is True and i % 10 == 0:
            m1 = 'Log Likelihood after {}'.format(i)
            m2 = ' iterations: {}'.format(L.round(5))
            print(m1 + m2)

        prev_L = L
        pi, m, S = maximization(X, g)
        g, L = expectation(X, pi, m, S)

        i += 1

    if verbose is True:
        msg = 'Log Likelihood after {} iterations: {}'\
            .format(i, L.round(5))
        print(msg)

    return (pi, m, S, g, L)
