#!/usr/bin/env python3
""" contains the function create_placeholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """returns two placeholders"""
    x = tf.placeholder(tf.float32, [None, nx], 'x')
    y = tf.placeholder(tf.float32, [None, classes], 'y')
    return x, y
