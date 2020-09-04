#!/usr/bin/env python3
""" convolution graysclae SAME"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs SAME convolution on grayscale images.
    if necessary the image should be padded with 0s
    """
    kh, kw = kernel.shape
    pad_w = numpy.ceil(kw) if kw % 2 == 0 else numpy.floor(kw)
    pad_h = numpy.ceil(kh) if kh % 2 == 0 else numpy.floor(kh)

    padded = np.pad(images, (0, 0), (pad_h, pad_h), (pad_w, pad_w), 'constant')
    m, h, w = images.shape
    conv = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            sector = padded[:, i:i+kh, j:j+kw]
            s_c = sector * kernel
            p = np.sum(s_c, axis=1)
            p2 = np.sum(p, axis=1)
            conv[:, i, j] = p2

    return conv
