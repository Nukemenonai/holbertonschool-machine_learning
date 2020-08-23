#!/usr/bin/env python3
"""batch normalization upgraded module"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network
    in tensorflow
    prev: activated output of previous layer
    n: number of nodes in layer"""

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=w_init)
    Z = layers(prev)

    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)

    epsilon = tf.constant(1e-8)

    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)

    if not activation:
        return Z_norm
    else:
        A = activation(Z_norm)
        return A
