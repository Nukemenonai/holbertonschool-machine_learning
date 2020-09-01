#!/usr/bin/env python3
"""test model module"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network:
    data: input data to test the model with
    labels: correct one-hot labels of data
    verbose: boolean that determines if output should be printed
    during the testing process
    Returns: the loss and accuracy of the model"""
    loss, acc = network.evaluate(x=data, y=labels, verbose=verbose)
    return loss, acc
