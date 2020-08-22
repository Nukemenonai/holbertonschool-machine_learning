#!/usr/bin/env python3
"""contains the Adam optimization function"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using Adam algorithm"""
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    Sd = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    Vd_C = Vd / (1 - (beta1 ** t))
    Sd_C = Sd / (1 - (beta2 ** t))

    updated_var = var - (alpha * (Vd_C / (np.sqrt(Sd_C) + epsilon)))
    return var, Vd, Sd
