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
        for item in layers:
            if type(item) != int or item < 1:
                raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
