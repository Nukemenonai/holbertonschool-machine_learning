#!/usr/bin/env python3
""" this module contains the minor function"""


def determinant(matrix):
    """ Calculates the determinant of a matrix"""

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if len(matrix) == 1:
        if len(matrix[0]) == 0:
            return 1
        if len(matrix[0]) == 1:
            return matrix[0][0]

    for elem in matrix:
        if len(elem) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 2:
        d = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return d

    value = 0
    for i in range(len(matrix)):
        copy = [el[:] for el in matrix]
        copy = copy[1:]
        size = len(copy)

        for j in range(size):
            copy[j] = copy[j][0:i] + copy[j][i + 1:]
        cofactor = (-1) ** (i % 2)
        det = determinant(copy)
        value += cofactor * matrix[0][i] * det
    return value


def minor(matrix):
    """ calculates the minor matrix of a matrix """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix is [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    if not all(len(matrix) == col for col in [len(row) for row in matrix]):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]
    minors = [item[:] for item in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            minors[i][j] = determinant([r[:j]
                                        + r[j+1:]
                                        for r in (matrix[:i] + matrix[i+1:])])
    return minors
