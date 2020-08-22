#!/usr/bin/env python3
""" this module contains the update variables momentum function"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum algo
    alpha: learning rate
    beta1: momentum weight
    var: variable to be updated
    grad: gradient of var
    v: previous moment of var
    return: updated variable, new moment
    """
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    var_updated = var - (alpha * Vd)
    return var_updated, Vd
