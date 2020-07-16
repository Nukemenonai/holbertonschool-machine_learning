#!/usr/bin/env python3
"""contains the np_elementwise function """


def np_elementwise(mat1, mat2):
    """ performs element wise addition substraction
    multiplication and division"""
    return (mat1 + mat2, mat1 - mat2,
            mat1 * mat2, mat1 / mat2)
