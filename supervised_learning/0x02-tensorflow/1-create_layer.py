#!/usr/bin/env python3
"""  this module contains the create_layer function """


import tensorflow as tf


def create_layer(prev, n, activation):
    """ creates a layer :v """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.Dense(n, activation, True,
                           weights, name='layer').apply(prev)
