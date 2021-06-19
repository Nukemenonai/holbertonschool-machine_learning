#!/usr/bin/env python3
"""
t-SNE (stochastic Neightbor Embedding)
Shannon Entropy and P affinities
"""

import numpy as np


def HP(Di, beta):
    """
    * Di: numpy.ndarray of shape (n - 1,) containing the
      pariwise distances between a data point and all other
      points except itself
      - n: number of data points
    * beta: beta value for the Gaussian distribution
    Returns: (Hi, Pi)
    * Hi: the Shannon entropy of the points
    * Pi: numpy.ndarray of shape (n - 1,) containing the P
      affinities of the points
    """
    # original equation of P(ij)
    num = np.exp(-Di.copy() * beta)
    den = np.sum(np.exp(-Di.copy() * beta))
    Pi = num / den

    # equation of H(i)
    Hi = -np.sum(Pi * np.log2(Pi))

    return (Hi, Pi)
