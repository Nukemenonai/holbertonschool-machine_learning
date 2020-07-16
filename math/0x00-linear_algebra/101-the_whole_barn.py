#!/usr/bin/env python3
""" this file contains the add_matrices file"""


def matrix_shape(matrix):
    """ determines the shape of a matrix"""
    size = []
    mtrx = matrix
    size.append(len(matrix))
    if type(mtrx[0]) == list:
        while type(mtrx[0]) != int:
            size.append(len(mtrx[0]))
            mtrx = mtrx[0]
    return size


def add_matrices(mat1, mat2):
    """ adds matrices in 2 3 and 4 dimensions """
    if(type(mat1[0]) == int) and (len(mat1) == len(mat2)):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    if (matrix_shape(mat1) == matrix_shape(mat2)):
        if len(matrix_shape(mat1)) == 4:
            return([[[[mat1[i][j][k][l] + mat2[i][j][k][l]
                       for l in range(len(mat1[i][j][k]))]
                      for k in range(len(mat1[i][j]))]
                     for j in range(len(mat1[i]))]
                    for i in range(len(mat1))])

        elif len(matrix_shape(mat1)) == 3:
            return [[[mat1[i][j][k] + mat2[i][j][k]
                      for k in range(len(mat1[i][j]))]
                     for j in range(len(mat1[i]))]
                    for i in range(len(mat1))]

        elif len(matrix_shape(mat1)) == 2:
            return [[mat1[i][j] + mat2[i][j]
                     for j in range(len(mat1[i]))]
                    for i in range(len(mat1))]
    else:
        return None
