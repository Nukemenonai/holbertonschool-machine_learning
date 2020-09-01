#!/usr/bin/env python3
"""
saving and loading a model
"""

import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model:"""
    K.models.save_model(model=network,
                        filepath=filename)
    return None


def load_model(filename):
    """loads an entire model:"""
    model = K.load_model(filename)
    return model
