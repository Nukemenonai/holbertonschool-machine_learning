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
    intializer = K.initializers.he_normal()
    norm1 = K.layers.BatchNormalization(axis=3)(X)
    A = K.layers.Activation('relu')(norm1)
    conv = K.layers.Conv2D(filters=fltr,
                           kernel_size=(1, 1),
                           padding='same',
                           strides=(1, 1)
                           kernel_initializer=initializer)(A)
    avgpool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(conv)
    return avgpool, fltr
