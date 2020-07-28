#!/usr/bin/env python3
""" This module contains the Poisson class"""


e = 2.7182818285


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
            self.data = data
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """instance method.
        calculates the PMF value for a number of successes
        k represents the number of successes """
        if type(k) != int:
            int(k)
        if k < 0:
            return 0

        def factorial(n):
            """returns the factorial of given number """
            return 1 if (n == 1 or n == 0) else n * factorial(n - 1)

        return ((self.lambtha ** k)*(e ** -(self.lambtha)))/factorial(k)
