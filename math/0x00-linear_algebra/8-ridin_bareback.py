#!/usr/bin/env python3
""" this file contains the mat_mul function"""


def mat_mul(mat1, mat2):
    """ multiplies two matrices """
    if (len(mat1[0]) != len(mat2)):
        return None
    else:
        new = [[0 for cols in range(len(mat2[0]))]
               for rows in range(len(mat1))]
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat2)):
                    new[i][j] += mat1[i][k] * mat2[k][j]
    return new
