#!/usr/bin/env python3
""" This module contains the Poisson class"""


class Poisson:
    """ this class representas a Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """ class constructor """
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
