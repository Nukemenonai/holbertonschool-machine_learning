#!/usr/bin/env python3
"""specificity modula"""

import numpy as np


def specificity(confusion):
    """ calculates specificty for each class in a confussion matrix"""
    TP = np.diagonal(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    return TN / (TN + FP)
