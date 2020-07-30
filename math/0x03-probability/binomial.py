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
            self.mean = sum(data) / len(data)
            self.stddev = (sum(
                [(x - self.mean) ** 2 for x in data]
            ) / len(data)) ** 0.5
            self.variance = self.stddev ** 2
            self.p = 1 - (self.variance / self.mean)
            self.n = int(round((self.mean * 1) / self.p))
            self.p = self.mean / self.n

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
        p2 = ((p ** k) * (q ** (n - k)))
        return p1 * p2

    def cdf(self, k):
        """calculates the value of the CDF for a number of successes """
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(k + 1)])
