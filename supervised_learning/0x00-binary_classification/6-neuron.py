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
        m = np.shape(Y)
        return -(1 / m[1]) * np.sum(Y * np.log(A) +
                                    (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """ Evaluates the neurons predictions
        X: contains the input data
        Y: np.nadarray(1, m) contains the correct labels"""
        Z = self.forward_prop(X)
        return np.rint(Z), self.cost(Y, Z)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ calculates one pass of the gradient descent on the neuron
        X: nd array (nx, m) contains input data
        Y: nd array (1, m) contains correct lables
        A: nd array (1, m) contains activated output
        alpha: learning rate
        """
        XT = X.transpose()
        m = np.shape(Y)
        loss = A - Y
        gradient = np.dot(loss, XT) / m[1]
        self.__W = self.__W - alpha * gradient
        db = loss.sum() / m[1]
        self.__b = self.__b - alpha * db
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ trains the neuron
        X: ndarray(nx, m)
        Y: correct labels
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.__W, self.__b = self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
