#!/usr/bin/env python3
"""this module contains the Exponential distribution class"""


e = 2.7182818285


class Exponential:
    """ represents  exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """ class constructor """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.data = data
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """ calculates the value of the PDF
        for a given period of time """
        if x < 0:
            return 0
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """ Calculates the CDF for a given time period
        x is the time period"""
        if x < 0:
            return 0
        return 1 - (e ** (-self.lambtha * x))
