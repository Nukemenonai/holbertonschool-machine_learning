#!/usr/bin/env python3
""" this file contains the cat_matrices function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates along a specific axis  """
    cp1 = [[col for col in row] for row in mat1]
    cp2 = [[col for col in row] for row in mat2]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        new = cp1 + cp2
        return new
    if axis == 1 and (len(mat1) == len(mat2)):
        for row in mat1:
            if len(row) != len(mat1[0]):
                return None
        new2 = [cp1[i] + cp2[i] for i in range(len(cp2))]
        return new2
    return None
