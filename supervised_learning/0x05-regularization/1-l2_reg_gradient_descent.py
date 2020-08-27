#!/usr/bin/env python3
""" gradient descent with l2"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """calculates gradient descent using l"""
    dZ = []
    m = np.shape(Y)[1]
    dZ.append(cache['A' + str(L)] - Y)
    for l in range(L, 0, -1):
        A = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        dg = 1 - (A ** 2)
        dWdx = np.matmul(dZ[L - l], A.T) / m
        dbdx = np.sum(dZ[L - l], axis=1, keepdims=True) / m
        dzdx = dZ.append(np.matmul(W.T, dZ[L - l]) * dg)
        l2 = dWdx + (lambtha / m) * weights['W' + str(l)]
        weights['W' + str(l)] -= alpha * l2
        weights['b' + str(l)] -= alpha * dbdx
