#!/usr/bin/env python3
"""
backward propagation with pooling
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs backward propagation over a pool on a NN"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for el in range(m):
        im = A_prev[el]
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    strided_h = i * sh
                    strided_w = j * sw
                    end_h = strided_h + kh
                    end_w = strided_w + kw
                    X = im[strided_h:end_h, strided_w:end_w, k]
                    if mode == 'max':
                        mask = np.where(X == np.max(X), 1, 0)
                        aux = mask * dA[el, i, j, k]
                        dA_prev[el, strided_h:end_h, strided_w:end_w, k] += aux
                    if mode == 'avg':
                        avg = dA[el, i, j, k] / (kh * kw)
                        mask = np.ones(X.shape)
                        result = mask * avg
                        dA_prev[el, strided_h:end_h,
                                strided_w:endw, k] += result
    return dA_prev
