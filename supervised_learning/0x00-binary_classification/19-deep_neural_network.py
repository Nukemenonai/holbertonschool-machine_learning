#!/usr/bin/env python3
"""this module contains the a deep neural network class"""

import numpy as np


def sigmoid(a):
    """ returns the sigmoid activation """
    return 1/(1 + np.exp(-a))


class DeepNeuralNetwork:
    """defines a deep neural network"""
    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) != int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights['W' + str(i + 1)] = (np.random.randn(layers[i], nx) *
                                              np.sqrt(2./nx))
            nx = layers[i]
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ returns the L privatized property"""
        return self.__L

    @property
    def cache(self):
        """return the privatized cache property """
        return self.__cache

    @property
    def weights(self):
        """returns the privatized weight property """
        return self.__weights

    def forward_prop(self, X):
        """calculates the forward propagation of a deep neural network"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            Z = np.matmul(W, self.__cache['A' + str(i)]) + b
            self.__cache['A' + str(i + 1)] = sigmoid(Z)
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        m = np.shape(Y)
        return -(1 / m[1]) * np.sum(Y * np.log(A) +
                                    (1 - Y) * np.log(1.0000001 - A))
