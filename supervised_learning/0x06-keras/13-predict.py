#!/usr/bin/env pytho
""" prediction module"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes a prediction using a neural network:
    network: model to make the prediction with
    data: input data to make the prediction with
    verbose: boolean that determines if output
    should be printed during the prediction process
    Returns: the prediction for the data"""
    prediction = network.predict(x=data, verbose=verbose)
    return prediction
