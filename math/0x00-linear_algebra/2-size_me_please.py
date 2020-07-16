#!/usr/bin/env python3
""" returns the size of a matrix """


def matrix_shape(matrix):
    """ this code returns the size or shape of a matrix"""
    size = []
    mtrx = matrix
    size.append(len(matrix))
    if type(mtrx[0]) == list:
        while (type(mtrx[0]) != int) and (type(mtrx[0]) != float):
            size.append(len(mtrx[0]))
            mtrx = mtrx[0]
    return size
