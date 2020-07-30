#!/usr/bin/env python3
""" This module contains the normal distribution class"""


π = 3.1415926536
e = 2.7182818285


def sq(x):
    """ returns square root """
    return x ** 0.5


def erf(x):
    """ error function """
    return (2/sq(π)) * (x - (x**3/3) + (x**5/10) -
                        (x**7/42) + (x**9/216))


class Normal:
    """ represents a normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ class constructor """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = (sum(data) / len(data))
            self.stddev = (sum(
                [(x - self.mean) ** 2 for x in data]
            ) / len(data)) ** 0.5

    def z_score(self, x):
        """ calculates the zscore of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ calculates the x value of a z-score"""
        return (self.stddev * z) + self.mean

    def pdf(self, x):
        """ Calculates the value of PDF for a given x-value"""
        sigma = self.stddev
        mu = self.mean
        variance = sigma ** 2
        p1 = 1 / (sigma * sq(2 * π))
        p2 = e ** ((-1/2) * ((x - mu)/sigma) ** 2)
        return p1 * p2

    def cdf(self, x):
        """ Calculates de value of CDF for a given x value """
        mu = self.mean
        sigma = self.stddev
        p2 = (x - mu)/(sigma * sq(2))
        p3 = 1 + erf(p2)
        return 1/2 * p3
