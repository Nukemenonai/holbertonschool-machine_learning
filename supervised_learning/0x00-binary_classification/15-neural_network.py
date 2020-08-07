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

    def evaluate(self, X, Y):
        """evaluates the neural networks predictions"""
        Z, Z2 = self.forward_prop(X)
        return np.where(Z2 < 0.5, 0, 1), self.cost(Y, Z2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass of the gradient descent"""
        m = np.shape(Y)
        dz2 = A2 - Y
        dW2 = np.matmul(dz2, A1.transpose()) / m[1]
        db2 = np.sum(dz2, axis=1, keepdims=True) / m[1]
        dz1 = np.matmul(self.__W2.transpose(), dz2) * (A1 * (1 - A1))
        dW1 = np.matmul(dz1, X.transpose()) / m[1]
        db1 = np.sum(dz1, axis=1, keepdims=True) / m[1]
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains the neural network"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose == True or graph == True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost, iters = [], []
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            cst = self.cost(Y, A2)
            if verbose == True:
                if i % step == 0:
                    print("Cost after {} iterations: {}".format(i, cst))
                    cost.append(cst)
                    iters.append(i)
        if graph == True:
            plt.plot(cost, iters)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()
        return self.evaluate(X, Y)