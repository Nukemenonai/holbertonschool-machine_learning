#!/usr/bin/env python3
""" this module contains the poly_derivative function"""


def poly_derivative(poly):
    """ calculates the derivative of a polinomial """
    if type(poly) != list:
	return None
    for item in poly:
        if type(item) != int:
            return None
    deriv = [0]
    for i in range(len(poly)):
        if i == 0:
            continue
        else:
            deriv.append(poly[i] * (i))
    if deriv[0] == 0 and len(deriv) != 1:
        del deriv[0]
    return deriv
