#!/usr/bin/env python3
""" saving and loading weights"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model’s weights:
    network is the model whose weights should be saved
    filename is the path of the file that the weights should be saved to
    save_format is the format in which the weights should be saved
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads a model’s weights:
    network is the model to which the weights should be loaded
    filename is the path of the file that the weights should be loaded from
    """
    network.load_weights(filepath=filename)
    return None
