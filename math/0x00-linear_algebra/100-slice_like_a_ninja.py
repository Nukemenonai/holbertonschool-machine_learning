#!/usr/bin/env python3
""" :v """


def np_slice(matrix, axes={}):
    """ :v """
    tuplas = {
        0: (None, None, None),
        1: (None, None, None),
        2: (None, None, None)
    }
    for key in axes.keys():
        tuplas[key] = axes[key]
    if matrix.ndim == 2:
        return matrix[slice(*tuplas[0]), slice(*tuplas[1])]
    if matrix.ndim == 3:
        return matrix[slice(*tuplas[0]), slice(*tuplas[1]), slice(*tuplas[2])]
