#!/usr/bin/env python3
"""dense block."""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dese block as described in densely
    connected convolutional networks
    X: output from previous layer
    nb_filters: number of filters in X
    growth rate for the dense block
    layers: number of layers of the dense block
    """
    initializer = K.initializers.he_normal()

    for i in range(layers):
        norm1 = K.layers.BatchNormalization(axis=3)(X)
        A1 = K.layers.Activation('relu')(norm1)
        bottleneck = K.layers.Conv2D(filters=(4 * growth_rate),
                                     kernel_size=(1, 1),
                                     padding='same',
                                     strides=(1, 1),
                                     kernel_initializer=initializer)(A1)
        norm2 = K.layers.BatchNormalization(axis=3)(bottleneck)

        A2 = K.layers.Activation('relu')(norm2)

        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding='same',
                                strides=(1, 1),
                                kernel_initializer=initializer)(A2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate
    return (X, nb_filters)
