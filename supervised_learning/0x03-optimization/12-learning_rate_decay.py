#!/usr/bin/env python3
"""learning rate decay module"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates the learning rate decay operation using tensorflow"""
    return tf.train.inverse_time_decay(alpha,
                                       global_step,
                                       decay_step,
                                       decay_rate,
                                       staircase=True)
