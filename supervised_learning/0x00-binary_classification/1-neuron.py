#!/usr/bin/env python3
"""This file contains the Privatized class Neuron """
import numpy as np


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
