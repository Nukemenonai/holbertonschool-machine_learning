#!/usr/bin/env python3
"""Regular chains"""

import numpy as np


def regular(P):
    """
    determines the steady state probabilities
    of a regular markov chain:
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the transition matrix
        P[i, j] is the probability of transitioning
        from state i to state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n)
    containing the steady state probabilities, None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    if P.shape[0] != P.shape[1]:
        return None

    sum_t = np.sum(P, axis=1)
    for item in sum_t:
        if not np.isclose(item, 1):
            return None

    _, eig_vec = np.linalg.eig(P.T)

    normalization = (eig_vec/eig_vec.sum()).real

    a = np.dot(normalization.T, P)

    for item in a:
        if (item >= 0).all() and np.isclose(item.sum(), 1):
            return item.reshape(1, -1)
    return None
