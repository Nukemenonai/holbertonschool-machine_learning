#!/usr/bin/env python3
""" pool forward """

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over
    a pooling layer of a neural network: """
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    kh, kw = kernel_shape

    pool_height = int(((h_prev - kh) / sh) + 1)
    pool_width = int(((w_prev - kw) / sw) + 1)

    pool = np.zeros((m, pool_height, pool_width, c_prev))

    for i in range(pool_height):
        for j in range(pool_width):
            strided_h = i * sh
            strided_w = j * sw
            end_h = strided_h + kh
            end_w = strided_w + kw
            x = A_prev[:, strided_h:end_h, strided_w:end_w]
            if mode == 'max':
                Wx = x.max(axis=(1, 2))
            elif mode == 'avg':
                Wx = x.mean(axis=(1, 2))
            pool[:, i, j] = Wx
    return pool
