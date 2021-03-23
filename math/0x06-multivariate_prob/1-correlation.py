#!/usr/bin/env python3
""" this module finds the correlation matrix"""


import numpy as np


def correlation(C):
    """ calculate a correlation matrix
    :param C: matrix of covariances
    :return: np.ndarray with 
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')

    if len(C.shape) is not 2 or C.shape[0] is not C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    D = np.sqrt(np.diag(C))
    outer = np.outer(D, D)
    return C / outer
