#!/usr/bin/env python3
""" """

import numpy as np


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes l2 regularization"""

    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    return tf.layers.Dense(n, activation, True,
                           weights, name='layer').apply(prev)
