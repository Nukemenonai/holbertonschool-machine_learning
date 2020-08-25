#!/usr/bin/env python3
"""f1 score module """


import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the f1 score of a confusion matrix"""
    TPR = sensitivity(confusion)
    PPV = precision(confusion)
    F1 = 2 * ((PPV * TPR)/(PPV + TPR))
    return F1
