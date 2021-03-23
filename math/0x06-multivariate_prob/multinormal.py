#!/usr/bin/env python3
"""
this module contains the Multinormal class
"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set
    :param X: numpy.darray
    :return: mean, cov
    """
    d = X.shape[0]
    n = X.shape[1]

    mean = np.mean(X, axis=1).reshape(d, 1)
    X = X - mean

    cov = ((np.dot(X, X.T)) / (n - 1))

    return mean, cov


class MultiNormal ():
    """ Class that represents a Multivariate Normal distribution """
    def __init__(self, data):
        """
        init class
        :param data: np.ndarray
            n is the number of data points
            d number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) is not 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean, self.cov = mean_cov(data)

    def pdf(self, x):
        """ calculates the PDF at a data point
        :param x: np.ndarray contains the data set
            d number of dimensions of the multinomial instance
        :return: the value of the PDF
        """

        if type(x) != np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError(f"x must have the shape ({d}, 1 )")

        x_1 = x - self.mean

        den = np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        exp = np.exp(-(np.linalg.solve(self.cov, x_1).T.dot(x_1)) / 2)
        res = (1/den) * exp
        return res[0][0]
