#!/usr/bin/env python3
"""this module contains the a deep neural network class"""

import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network"""
    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if type(layers[i]) != int:
                raise TypeError("layers must be a list of positive integers")
            self.weights['W' + str(i + 1)] = (np.random.randn(layers[i], nx) *
                                              np.sqrt(2./nx))
            nx = layers[i]
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
