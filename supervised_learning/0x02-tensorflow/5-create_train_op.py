#!/usr/bin/env python3
""" creates the training operation """

import tensorflow as tf


def create_train_op(loss, alpha):
    """ creates the training operation from the network"""
    return tf.GradientDescentOptimizer(alpha).minimize(loss)

