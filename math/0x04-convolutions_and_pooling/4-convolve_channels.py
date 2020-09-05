#!/usr/bin/env python3
""" convolution grayscale """

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    pad_h, pad_w = (0, 0)

    if padding == 'same':
        pad_w = int((((w - 1) * sw + kw - w) / 2) + 1)
        pad_h = int((((h - 1) * sh + kh - h) / 2) + 1)

    if type(padding) == tuple and len(padding) == 2:
        pad_h, pad_w = padding

    padded = np.pad(images, ((0, 0), (pad_h, pad_h),
                             (pad_w, pad_w), (0, 0)), 'constant')

    output_h = ((h + (pad_h * 2) - kh) // sh) + 1
    output_w = ((w + (pad_w * 2) - kw) // sw) + 1

    conv = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            sector = padded[:, (i * sh):(i * sh) + kh,
                            (j * sw):(j * sw) + kw]
            s_c = sector * kernel
            p = np.sum(s_c, axis=1)
            p2 = np.sum(p, axis=1)
            p3 = np.sum(p2, axis=1)
            conv[:, i, j] = p3
    return conv
