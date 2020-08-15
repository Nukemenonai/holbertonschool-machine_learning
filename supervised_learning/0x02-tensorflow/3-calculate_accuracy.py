#!/usr/bin/env python3
"""this module contains the calculate accuracy function """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    acc = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(acc, tf.float32))
