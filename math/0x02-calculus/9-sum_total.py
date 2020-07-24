#!/usr/bin/env python3
""" this module contains the sumation i squared function """


def summation_i_squared(n):
    """ this function calculates the sum notation from i=1 to n """
    if type(n) != int:
        return None
    elif n <= 0:
        return None
    else:
        return int((n * (n + 1) * ((2 * n) + 1)) / 6)
