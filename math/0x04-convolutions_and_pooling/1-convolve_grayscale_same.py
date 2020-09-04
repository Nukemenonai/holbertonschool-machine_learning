#!/usr/bin/env python3
""" convolution graysclae SAME"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs SAME convolution on grayscale images.
    if necessary the image should be padded with 0s
    """
    kh, kw = kernel.shape
    pad_w = int(kw / 2) if kw % 2 == 0 else int((kw - 1) / 2)
    pad_h = int(kh / 2) if kh % 2 == 0 else int((kh - 1) / 2)

    padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant')
    m, h, w = images.shape
    conv = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            sector = padded[:, i:i+kh, j:j+kw]
            s_c = sector * kernel
            p = np.sum(s_c, axis=1)
            p2 = np.sum(p, axis=1)
            conv[:, i, j] = p2
    return conv
