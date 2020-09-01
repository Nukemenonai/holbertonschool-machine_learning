#!/usr/bin/env python3
"""trais a model using mini batch gradient descent"""

def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ trains a model using mini batch gradient descent"""
    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle)
    return history
