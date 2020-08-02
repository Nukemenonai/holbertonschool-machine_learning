#!/usr/bin/env python3
"""This file contains the Privatized class Neuron """
import numpy as np


def sigmoid(a):
    """ returns the sigmoid activation """
    return 1/(1 + np.exp(-a))


class Neuron:
    """defines a single neuron performing binary clasification"""
    def __init__(self, nx):
        """class constructor """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
        self.nx = nx

    @property
    def W(self):
        """getter for the weight attribute """
        return self.__W

    @property
    def b(self):
        """getter for the bias attribute """
        return self.__b

    @property
    def A(self):
        """getter for the Activated output attribute """
        return self.__A

    def forward_prop(self, X):
        """ calculates the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """ calculates the cost of the model using logistic regression"""
        one = 1.0000001
        m = np.shape(Y[0])
        m = m[0]
        J = - (1 / m) * (Y * np.log(A) + (1 - Y) * (np.log(one - A))).sum()
        return J
