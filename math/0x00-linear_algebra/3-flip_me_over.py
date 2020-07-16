#!/usr/bin/env python3
""" this file contains the matrix_transpose function """


def matrix_transpose(matrix):
    """ this code transposes a matrix """
    new = [[matrix[j][i] for j in range(len(matrix))]
           for i in range(len(matrix[0]))]
    return new
