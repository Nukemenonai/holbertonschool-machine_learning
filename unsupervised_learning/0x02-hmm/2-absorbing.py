#!/usr/bin/env python3
"""Absorbing markov chain"""

import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing

    Args:
        P: standard transition matrix
    Return: True if it's absorbing, None on failure
    """
    if (P == np.eye(P.shape[0])).all():
        return True
    if np.any(np.diag(P) == 1):
        for i, row in enumerate(P):
            for j, col in enumerate(row):
                if i == j and ((i + 1) < len(P)) and ((j + 1) < len(P)):
                    if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                        return False
        return True
    return False
