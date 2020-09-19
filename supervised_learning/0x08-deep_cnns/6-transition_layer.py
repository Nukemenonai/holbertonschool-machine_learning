#!/usr/bin/env python3
"""transition layer"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """builds a transition layer
    X: input form the previous layer
    nb_filters: number of filters in X
    compression: compression factor of transition layer
    """
    fltr = int(nb_filters * compression)
    norm1 = K.layers.BatchNormalization(axis=3)(X)
    A = K.layers.BatchNormalization('relu')(norm1)
    conv = K.layers.Conv2D(filters=filter,
                           kernel_size(1, 1),
                           padding='same',
                           strides=(1, 1))(A)
    avgpool = K.layers.AeragePooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(conv)
    return avgpool, fltr
