#!/usr/bin/env python3

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    acc = tf.equal(tf.argmax(y), tf.argmax(y_pred))
    return tf.reduce_mean(tf.cast(acc, tf.float32))
