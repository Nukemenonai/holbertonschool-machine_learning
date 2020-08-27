#!/usr/bin/env python3
"""L2 regularization cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network
    with L2 regularization"""

    norm = 0
    for i in range(1, L + 1):
        idx = 'W' + str(i)
        W = np.linalg.norm(weights[idx])
        norm += W

    return cost + (lambtha / (2 * m)) * norm
