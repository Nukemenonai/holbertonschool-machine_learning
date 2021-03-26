#!/usr/bin/env python3
""" marginal """

import numpy as np


def marginal(x, n, P, Pr):
    """
    calculates the marginal probability of obtaining the data:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    nf = np.math.factorial(n)
    xf = np.math.factorial(x)
    nxf = np.math.factorial(n - x)

    L = (nf / (xf * nxf)) * (P ** x) * ((1 - P) ** (n - x))

    # P(AnB) = P(A) * P(B|A)
    # marginal likelihood = evidence
    return np.sum(Pr * L)
