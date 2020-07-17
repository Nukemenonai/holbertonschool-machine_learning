#!/usr/bin/env python3
""" this file contains the np slice function """


def np_slice(matrix, axes={}):
    """ slices a matrix along a specific axis """
    tuplas = {}
    highest = max(axes.keys())
    for i in range(highest):
        tuplas[i] = (None, None, None)
    for key in axes.keys():
        tuplas[key] = axes[key]

    return matrix[tuple(slice(*i) for i in tuplas.values())]
