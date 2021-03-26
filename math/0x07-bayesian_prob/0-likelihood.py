#!/usr/bin/env python3
""" likelihood """

import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given 
    various hypothetical probabilities 
    of developing severe side effects
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical 
    probabilities of developing severe side effects
    """
    if n <= 0: 
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise ValueError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    
    nf = np.math.factorial(n)
    xf = np.math.factorial(x)
    nxf = np.math.factorial(n - x)

    L = (nf / (xf * nxf)) * (P ** x) * ((1 - P) ** (n - x))

    return L 