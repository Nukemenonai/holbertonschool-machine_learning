#!/usr/bin/env python3
""" backward propagation with convolution"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ performs forward propagation over
    a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    if padding == "valid":
        ph = pw = 0

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    dA_pad = np.pad(A_pad, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    for el in range(m):
        im = A_pad[el]
        dIm = dA_pad[el]
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    strided_h = i * sh
                    strided_w = j * sw
                    end_h = strided_h + kh
                    end_w = strided_w + kw
                    x = im[strided_h:end_h, strided_w:end_w]
                    aW = W[:, :, :, k] * dZ[el, i, j, k]
                    dIm[strided_h: end_h, strided_w:end_w] += aW
                    dW[:, :, :, k] += x * dZ[el, i, j, k]

        if padding == "valid":
            dA[el] += dIm
        elif pdding == "same":
            dA[el] += dIm[ph: -ph, pw: -pw]

    return (dA, dW, db)
