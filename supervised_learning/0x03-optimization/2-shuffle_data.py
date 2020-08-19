#!/usr/bin/env python3

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    xC = np.copy(X)
    yC = np.copy(Y)
    assert len(xC) == len(yC)
    r = np.random.permutation(len(xC))
    return xC[r],yC[r]
