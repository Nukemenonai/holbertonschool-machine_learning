#!/usr/bin/env python3
"""dropout forward propagation"""

import numpy as np

def tanh(a):
    """ returns a tanh activation"""
    return np.tanh(a)


def softmax(a):
    """returns the softmax activation function"""
    return np.exp(a)/np.sum(np.exp(a), axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """calculates the forward propagation of a deep neural network
    using dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(L):
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = np.matmul(W, cache['A' + str(i)]) + b
        if i != L - 1:
            cache['A' + str(i + 1)] = tanh(Z)
            cache['D' + str(i + 1)] = np.random.binomial(1, keep_prob,
                                                         size=Z.shape)
            A = np.multiply(cache['A' + str(i + 1)], cache['D' + str(i + 1)])
            cache['A' + str(i + 1)] = A / keep_prob
        else:
            cache['A' + str(i + 1)] = softmax(Z)
    return cache
