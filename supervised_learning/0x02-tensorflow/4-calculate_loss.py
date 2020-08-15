#!/usr/bin/env python3
""" calculates loss :v"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """ calculates the loss :v """
    return tf.losses.softmax_cross_entropy(y, y_pred)
