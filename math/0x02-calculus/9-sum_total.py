#!/usr/bin/env python3
""" this module contains the sumation i squared function """


def summation_i_squared(n):
    """ this function calculates the sum notation from i=1 to n """
    res = 0
    for i in range(1, n+1):
        res += i**2
    return res
