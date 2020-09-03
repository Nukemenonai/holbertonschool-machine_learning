#!/usr/bin/env python3
""" convolution graysclae"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ performas valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = (h - kh +1)
    output_w = (w - kw +1)
    conv = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            sector = images[:, i:i+kh, j:j+kw]
            s_c = sector * kernel
            print(s_c)
            p = np.sum(s_c)
            p2 = np.sum(p)
            conv[:,i,j] = p2

    return conv
