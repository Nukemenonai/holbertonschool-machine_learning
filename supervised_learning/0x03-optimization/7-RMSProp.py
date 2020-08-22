#!/usr/bin/env python3
""" this module contains the RMS update function """

import numpy as np

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using RMSProp optimization algo
    alpha: learning rate
    beta2: RMSProp weight 
    epsilon: small number to avoid division by zero
    var: numpy.ndarray with variable to be updated
    grad: gradient of var (np.ndarray)
    s: previous second moment of var
    Return: updated var  and new moment. 
    """
    Sd = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var_updated = var - ((alpha * grad) / (np.sqrt(Sd) + epsilon))
    return var_updated, Sd
