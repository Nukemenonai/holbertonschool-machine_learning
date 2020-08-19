#!/usr/bin/env python3
"""contains the normalization constants function"""


import numpy as np


def normalization_constants(X):
    """calculates the standarization constants of a matrix"""
    mean = np.sum(X, axis=0) / X.shape[0]
    variance = np.sum((X - mean) ** 2, axis=0) / X.shape[0]
    std = np.sqrt(variance)
    return mean, std
