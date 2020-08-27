#!/usr/bin/env python3
""""""


import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network
    with l2 regularization"""
    return cost.get_regularization_losses()
