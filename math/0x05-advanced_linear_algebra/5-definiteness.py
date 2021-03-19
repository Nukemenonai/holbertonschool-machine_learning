#!/usr/bin/env python3
""" this module contains the function to find the definiteness of a matrix"""

import numpy as np

def definiteness(matrix):
    """ calculates the definiteness of a matrix"""

    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.linalg.eig(matrix):
        return None
    if not (matrix.transpose() == matrix).all():
        return None

    w, v = np.linalg.eig(matrix)

    pos = neg = zero = 0
    for eig_val in w:
        if eig_val == 0:
            zero += 1
        elif eig_val < 0:
            neg += 1
        else: 
            pos += 1
    
    if pos and not neg and not zero:
        return 'Positive definite'
    elif pos and not neg and zero:
        return 'Positive semi-definite'
    elif neg and not pos and zero:
        return 'Negative semi-definite'
    elif neg and not pos and not zero:    
        return 'Negative definite'
    return 'Indefinite'
