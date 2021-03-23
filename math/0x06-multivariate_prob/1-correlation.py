#!/usr/bin/env python3
""" this module finds the correlation matrix"""


import numpy as np


def correlation(C):
    """ calculates a correlation matrix
    C: matrix of covariances
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    n, d = C.shape

    if len(C.shape) is not 2 or C.shape[0] is not C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # extract the variances from the diagonal elements
    D = np.sqrt(np.diag(C))
    # get outer product of diagonal 
    outer = np.outer(D, D)
    return C / outer
