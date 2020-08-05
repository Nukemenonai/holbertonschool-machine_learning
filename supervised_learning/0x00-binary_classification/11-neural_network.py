#!/usr/bin/env python3
""" this module contains the neural network model"""

import numpy as np


def sigmoid(a):
    """ returns the sigmoid activation """
    return 1/(1 + np.exp(-a))


class NeuralNetwork:
    """ Neural neetwork with one hidden layer
    performing binary classification"""
    def __init__(self, nx, nodes):
        """class constructor """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """returns private W1"""
        return self.__W1

    @property
    def b1(self):
        """returns private b1 """
        return self.__b1

    @property
    def A1(self):
        """ returns private A1"""
        return self.__A1

    @property
    def W2(self):
        """returns private W2"""
        return self.__W2

    @property
    def b2(self):
        """ returns self b2"""
        return self.__b2

    @property
    def A2(self):
        """return self A2 """
        return self.__A2

    def forward_prop(self, X):
        """ calculates forward propagation of the NN"""
        Z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(Z)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        m = np.shape(Y)
        return -(1 / m[1]) * np.sum(Y * np.log(A) +
                                    (1 - Y) * np.log(1.0000001 - A))
    
