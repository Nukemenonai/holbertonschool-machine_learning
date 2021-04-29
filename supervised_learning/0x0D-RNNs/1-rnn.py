#!/usr/bin/env python3
"""
forward propagation of a single RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    * rnn_cell: instance of RNNCell that will be used for
      the forward propagation
    * X: data to be used, given as a numpy.ndarray of
      shape (t, m, i)
    * * t: maximum number of time steps
    * * m: batch size
    * * i: dimensionality of the data
    * h_0: initial hidden state, given as a numpy.ndarray
      of shape (m, h)
    * h: dimensionality of the hidden state
    Returns: H, Y
    * H: numpy.ndarray containing all of the hidden states
    * Y: numpy.ndarray containing all of the outputs
    """
    t, _, i = X.shape
    m, h = h_0.shape

    # saving the h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0
    h_prev = h_0

    for i in range(t):
        x_t = X[i]
        h_prev, y_pred = rnn_cell.forward(h_prev, x_t)
        H[i + 1] = h_prev
        Y[i] = y_pred
    return H, Y
