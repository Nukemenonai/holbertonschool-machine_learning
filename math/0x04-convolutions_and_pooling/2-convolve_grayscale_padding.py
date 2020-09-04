#!/usr/bin/env python3
""" convolution graysclae SAME"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs SAME convolution on grayscale images.
    if necessary the image should be padded with 0s
    """
    pad_h, pad_w = padding
    kh, kw = kernel.shape
    padded = np.pad(images, ((0, 0), (pad_h, pad_h),
                             (pad_w, pad_w)), 'constant')
    m, h, w = images.shape
    output_h = (h + (ph * 2) - kh + 1)
    output_w = (w + (pw * 2) - kw + 1)
    conv = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            sector = padded[:, i:i+kh, j:j+kw]
            s_c = sector * kernel
            p = np.sum(s_c, axis=1)
            p2 = np.sum(p, axis=1)
            conv[:, i, j] = p2
    return conv
