#!/usr/bin/env python3
""" contans the RMSProp updated module """


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation using tensorflow
    loss: loss of network
    alpha: learning rate
    beta2: RMSProp weight
    epsilon: to avoid division by zero
    Return: traning op """
    return tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss=loss)
