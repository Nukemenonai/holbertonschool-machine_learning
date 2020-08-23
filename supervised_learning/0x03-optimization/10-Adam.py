#!/usr/bin/env python3
"""contains the Adam optimization function"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creates Adam optimization operation with tensorflow"""
    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1,
                                  beta2=beta2,
                                  epsilon=epsilon).minimize(loss)
