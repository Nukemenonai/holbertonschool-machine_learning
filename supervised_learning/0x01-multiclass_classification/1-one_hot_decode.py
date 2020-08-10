#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
    """ converts a one-hot matrix to a vector of labels
    one-hot: one-hot encoded numpy array
    """
    decode = np.argmax(one_hot, axis=0)
    return decode
