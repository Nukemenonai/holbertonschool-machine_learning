#!/usr/bin/env python3
"""l2_reg_cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network
    with l2 regularization"""
    cost_l2 = tf.losses.get_regularization_losses(scope=None)
    return cost + cost_l2
