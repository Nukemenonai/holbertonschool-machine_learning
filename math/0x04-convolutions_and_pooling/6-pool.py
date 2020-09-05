#!/usr/bin/env python3
""" convolution on images with multiple kernels"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernels_shape
    sh, sw = stride

    output_h = int(((h - kh) / sh) + 1)
    output_w = int(((w - kw) / sw) + 1)
    conv = np.zeros((m, output_h, output_w, c))
    for i in range(output_h):
        for j in range(output_w):
            sector = images[:, (i * sh):(i * sh) + kh,
                            (j * sw):(j * sw) + kw]
            s_c = sector * kernels[:, :, :, k]
            if mode == 'max':
                p = np.max(sector, axis=1)
                p2 = np.max(p, axis=1)
            if mode == 'avg':
                p = np.mean(sector, axis=1)
                p2 = np.mean(p, axis=1)
            conv[:, i, j] = p2
    return conv
