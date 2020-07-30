#!/usr/bin/env python3
""" This module contains the Binomial distribution class """



def factorial(n):
    """returns the factorial of given number """
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Binomial:
    """ represents a binomial distribution """
    def __init__(self, data=None, n=1, p=0.5):
        """ class constructor """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.p = float(sum([i / 50 for i in data]) / (100))
            self.n = 50

    def pmf(self, k):
        """ calculates the PMF for a given number of successes"""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        n = self.n
        p = self.p
        q = 1 - p
        p1 = factorial(n) / (factorial(k) * factorial(n - k))
        p2 = ((p ** k) * (q ** (n -k)))
        return p1 * p2
