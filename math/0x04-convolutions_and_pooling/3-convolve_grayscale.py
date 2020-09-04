#!/usr/bin/env python3
""" convolution grayscale """

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ performs convolution on grayscale images.
    if necessary the image should be padded with 0s
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    pad_h, pad_w = (0, 0)
    if padding == 'same':
        pad_w = int((((iw - 1) * sw + kw - w) / 2) + 1)
        pad_h = int((((ih - 1) * sh + kh - w) / 2) + 1)
    elif type(padding) == tuple:
        pad_h, pad_w = padding

    padded = np.pad(images, ((0, 0), (pad_h, pad_h),
                             (pad_w, pad_w)), 'constant')
    output_h = (((h + (pad_h * 2) - kh) // sh) + 1)
    output_w = (((w + (pad_w * 2) - kw) // sw) + 1)
    conv = np.zeros((m, output_h, output_w))
    for i in range(0, output_h):
        for j in range(0, output_w):
            sector = padded[:, (i * sh):(sh * i) + kh,
                            (j * sw):(sw * j) + kw]
            s_c = sector * kernel
            p = np.sum(s_c, axis=1)
            p2 = np.sum(p, axis=1)
            conv[:, i, j] = p2
    return conv
