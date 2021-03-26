#!/usr/bin/env python3
""" posterior """

from scipy import special


def posterior(x, n, p1, p2):
    """
    that calculates the posterior probability for the various
    hypothetical probabilities of developing
    severe side effects given the data

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    p1
    p2
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or p1 < 0  or p1 > 1:
        raise TypeError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or p2 < 0  or p2 > 1:
        raise TypeError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    Pab1 = special.btdtr(x + 1, n -x + 1, p1)
    Pab2 = special.btdtr(x + 1, n -x + 1, p2)

    Pab = Pab2 - Pab1
    return Pab

    