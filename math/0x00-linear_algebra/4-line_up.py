#!/usr/bin/env python3
""" this file contains the add_arrays function"""


def add_arrays(arr1, arr2):
    """ this function sums 2 arrays element wise"""
    if (len(arr1) != len(arr2)):
        return None
    else:
        new = [arr1[i] + arr2[i] for i in range(len(arr1))]
    return new
