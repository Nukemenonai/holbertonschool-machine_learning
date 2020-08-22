#!/usr/bin/env python3
""" contains the omentum upgraded function """

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates training operation in tensorflow
    using GD with momentum
    """
    return tf.train.MomentumOptimizer(learning_rate=alpha,
                                      momentum=beta1).minimize(loss=loss)
