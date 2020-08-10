#!/usr/bin/env python3
""" this module contains the function one_hot_encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric verctor into a one-hot matrix
    m: number of examples
    maximum number of classes found in Y
    """
    encoding = np.zeros((classes, Y.shape[0]))
    for i in range(Y.shape[0]):
        if Y[i] > classes:
            return None
        encoding[Y[i],i] = 1
    return encoding
