#!/usr/bin/env python3
""" forward propagation with convolution"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over
    a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
    elif padding == 'valid':
        ph = pw = 0

    n_prev = np.pad(A_prev, (0, 0), (ph, ph), (pw, pw), (0, 0), 'constant')
    c_h = int(((h_prev + (ph * 2) - kh) / sh) + 1)
    c_w = int(((w_prev + (pw * 2) - kw) / sw) + 1)

    conv = np.zeros(m, c_h, c_w, c_new)

    for i in range(c_h):
        for j in range(c_w):
            for k in range(c_new):
                strided_h = i * sh
                strided_w = j * sw
                end_h = strided_h + kh
                end_w = strided_w + kw
                x = n_prev[:, strided_h:end_h, strided_w:end_w]
                Wx = W[:, :, :, k] * x
                Wx = Wx.sum(axis=(1, 2, 3))
                conv[:, i, j, k] = Wx

    Z = conv + b
    A = activation(Z)
    return A 
