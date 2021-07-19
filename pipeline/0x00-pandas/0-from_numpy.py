#!/usr/bin/env python3
"""
From Numpy
"""
import pandas as pd


def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray
    array: numpy array from which DF is created
    """
    columns = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    return pd.DataFrame(array, columns=columns[:len(array[1])])    