#!/usr/bin/env python3
"""precision module"""

import numpy as np


def precision(confusion):
    """calcultes the precision for each class
    in a confusion matrix"""
    TP = np.diagonal(confusion)
    FP_TP = np.sum(confusion, axis=0)
    return TP / FP_TP
