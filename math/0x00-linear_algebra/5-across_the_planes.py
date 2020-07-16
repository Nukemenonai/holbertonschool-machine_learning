#!/usr/bin/env python3
""" This file contains the matrix_shape function """


def add_matrices2D(mat1, mat2):
    """ adds 2 matrices """

    if len(mat2[0]) == len(mat1[0]) and len(mat1) == len(mat2):
        res = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
               for i in range(len(mat1))]
        return res
    else:
        return None
