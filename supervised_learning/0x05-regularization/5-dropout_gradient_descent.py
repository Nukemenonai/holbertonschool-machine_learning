#!/usr/bin/env python3
""" gradient descent with l2"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """calculates gradient descent using l"""
    dZ = []
    m = np.shape(Y)[1]
    dZ.append(cache['A' + str(L)] - Y)
    for ls in range(L, 0, -1):
        A = cache['A' + str(ls - 1)]
        W = weights['W' + str(ls)]
        dg = 1 - (A ** 2)
        dWdx = np.matmul(dZ[L - ls], A.T) / m
        dbdx = np.sum(dZ[L - ls], axis=1, keepdims=True) / m
        if ls != 1:
            reg_drop = dg * (cache['D' + str(ls - 1)] / keep_prob)
            dzdx = dZ.append(np.matmul(W.T, dZ[L - ls]) * dg)
        weights['W' + str(ls)] -= alpha * dWdx
        weights['b' + str(ls)] -= alpha * dbdx
