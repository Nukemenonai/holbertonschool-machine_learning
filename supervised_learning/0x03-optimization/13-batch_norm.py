#!/usr/bin/env python3
"""batch normalization module"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output with batch normalization"""
    batch_mean = np.sum(Z, axis=0) / Z.shape[0]
    batch_variance = (Z - batch_mean) ** 2
    batch_variance = np.sum(batch_variance, axis=0) / Z.shape[0]
    z_n = (Z - batch_mean) / np.sqrt(batch_variance + epsilon)
    return (gamma * z_n) + beta
