#!/usr/bin/env python3
"""Absorbing markov chain"""

import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing

    Args:
        P: standard transition matrix
    Return: True if it's absorbing, None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None

    n, _ = P.shape

    D = np.diag(P)

    if (D == 1).all():
        return True

    if (D != 1).all():
        return False

    absorb = np.where(D == 1, 1, 0)

    # we start to find connections from the absorbing states
    for i in range(n):
        idx = np.where(absorb == 1)

        for ind in idx[0]:
            item = P[:, ind]
            mask = np.where(item > 0)[0]
            absorb[mask] = True
            if absorb.all():
                return True

    return absorb.all()
