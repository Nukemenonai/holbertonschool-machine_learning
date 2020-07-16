#!/usr/bin/env python3
""" this file contains the np slice function """


def np_slice(matrix, axes={}):
    """ slices a matrix along a specific axis """
    tuplas = {
        0: (None, None, None),
        1: (None, None, None),
        2: (None, None, None),
        3: (None, None, None)
    }
    for key in axes.keys():
        tuplas[key] = axes[key]
    if matrix.ndim == 1:
        return matrix[slice(*tuplas[0])]
    if matrix.ndim == 2:
        return matrix[slice(*tuplas[0]), slice(*tuplas[1])]
    if matrix.ndim == 3:
        return matrix[slice(*tuplas[0]), slice(*tuplas[1]), slice(*tuplas[2])]
    if matrix.ndim == 4:
        return matrix[slice(*tuplas[0]), slice(*tuplas[1]),
                      slice(*tuplas[2]), slice(*tuplas[3])]
