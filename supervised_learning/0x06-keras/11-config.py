#!/usr/bin/env python3
""" save and load conf"""

import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    """
    settings = network.to_json()
    with open(filename, 'w') as f:
        f.write(settings)
    return None


def load_config(filename):
    """
    loads a model with a specific configuration:
    filename is the path of the file containing
    the model’s configuration in JSON format
    """
    with open(filename, 'r') as f:
        model = K.models.model_from_json(f.read())
    return model
