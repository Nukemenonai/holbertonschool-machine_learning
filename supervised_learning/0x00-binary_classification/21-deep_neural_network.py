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

    def evaluate(self, X, Y):
        """evaluates the network's predictions"""
        A, cache = self.forward_prop(X)
        return np.where(A < 0.5, 0, 1), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ calculates one pass of the gradient descent on NN"""
        m = np.shape(Y)[1]
        L = self.__L
        dZ = [cache['A' + str(L)] - Y]
        for l in range(L, 0, -1):
            A = cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            dg = (A * (1 - A))
            dWdx = np.matmul(dZ[L - l], A.T) / m
            dbdx = np.sum(dZ[L - l], axis=1, keepdims=True)
            dzdx = dZ.append(np.matmul(W.T, dZ[L - l]) * dg)
            self.__weights['W' + str(l)] -= alpha * dWdx
            self.__weights['b' + str(l)] -= alpha * dbdx
