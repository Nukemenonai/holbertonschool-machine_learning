#!/usr/bin/env python3
""" this module contains the poly_integral function"""


def poly_integral(poly, C=0):
    """ calculates the integral of a polinomial """
    if type(poly) != list or type(C) != int:
        return None
    for item in poly:
        if type(item) != int:
            return None
    if len(poly) == 0:
        return None
    integrate = [C]
    for i in range(len(poly)):
        integrate.append(poly[i] / (i + 1))
    return integrate
