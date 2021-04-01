#!/usr/bin/env python3
"""PDF"""

import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution:

    X: (n, d) the data points whose PDF should be evaluated
    m: (d,) the mean of the distribution
    S: (d, d) the covariance of the distribution
    return: P or None on failure
        P: (n,) the PDF values for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != S.shape[1] or S.shape[0] != S.shape[1]:
        return None
    if X.shape[1] != m.shape[0]:
        return None

    _, d = X.shape
    X_m = X - m
    S_i = np.linalg.inv(S)
    det = np.linalg.det(S)
    f = np.matmul(X_m, S_i)
    s = np.sum(f * X_m, axis=1)
    num = np.exp(s / -2)
    dn = np.sqrt(det) * ((2 * np.pi) ** (d / 2))
    pdf = num / dn

    P = np.where(pdf < 1e-300, 1e-300, pdf)

    # P = (1 / S_i * (np.sqrt(2 ** np.pi))) ** (0.5 * (X_m/S_i))
    return P
