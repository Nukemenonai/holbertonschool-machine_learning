#!/usr/bin/env python3
""" this file contains the np_cat function"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concatenates 2 matrices along a specific axis"""
    return np.concatenate((mat1, mat2), axis=axis)
