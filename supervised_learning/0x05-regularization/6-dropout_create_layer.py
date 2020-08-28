#!/usr/bin/env python3
"""l2_ reg create layer """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a tensorflow layer that includes l2 regularization"""

    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(keep_prob)
    return tf.layers.Dense(n, activation,
                           kernel_initializer=weights,
                           kernel_regularizer=reg).apply(prev)
