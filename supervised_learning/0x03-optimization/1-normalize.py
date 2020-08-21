#!/usr/bin/env python3
""" this module contains the normalize function"""

import numpy as np


def normalize(X, m, s):
    """calculates the standarization constants of a matrix"""
    Z_score = (X - m) / s
    return Z_score
