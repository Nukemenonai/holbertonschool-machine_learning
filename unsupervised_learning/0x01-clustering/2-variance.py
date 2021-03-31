#!/usr/bin/env python3
""" Variance """

import numpy as np


def variance(X, C):
    """
    calculates total intra cluster vairance for a data set
    X: np.ndarray, data set
    C: np.ndarray, centroid means for each cluster
    Return: var, total variance. None on failure.
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(C.shape) != 2:
            return None

        D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=-1))
        min_dist = np.min(D, axis=0)
        var = np.sum(min_dist ** 2)
        return var
    except Exception:
        return None
