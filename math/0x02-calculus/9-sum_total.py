#!/usr/bin/env python3
""" this module contains the sumation i squared function """


def summation_i_squared(n):
    """ this function calculates the sum notation from i=1 to n """
    if type(n) != int:
        return None
    if n == 1:
        return n
    else:
        return (n **2 ) + summation_i_squared(n-1)
