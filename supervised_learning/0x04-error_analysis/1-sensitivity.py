#!/usr/bin/env python3
"""sensitivity module"""


import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity
    of the confusion matrix"""
    TP_FN = np.sum(confusion, axis=1)
    TP = np.diagonal(confusion)
    return TP / TP_FN
