#!/usr/bin/env python3
"""
Policy Gradient
"""

import numpy as np


def policy(matrix, weight):
    """
    computes to policy with a weight of a matrix.
    matrix: matrix of states
    weights: the weights of the states
    """
    res = np.dot(matrix, weight)
    res = np.exp(res)

    return res / res.sum()


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient
    based on a state and a weight matrix.
    state: matrix representing the current
        observation of the environment
    weight: matrix of random weight
    return: the action and the gradient respectively
    """

    def softmax_grad(softmax):
        """computes jacobian of a softmax"""
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    probs = policy(state, weight)
    action = np.argmax(probs)

    d_softmax = softmax_grad(probs)[action, :]
    dlog = d_softmax / probs[0, action]
    gradient = state.T.dot(dlog[None, :])

    return (action, gradient)
