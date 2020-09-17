#!/usr/bin/env python3
"""
identity block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """builds an identity block
    A_prev: output from prevoius layer
    filters: number of filters in each convolution"""
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal()
    conv_F11 = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer)
    conv_F11 = conv_F11(A_prev)

    norm1 = K.layers.BatchNormalization(axis=3)
    norm1 = norm1(conv_F11)

    X1 = K.layers.Activation('relu')(norm1)

    conv_F3 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_initializer=initializer)
    conv_F3 = conv_F3(X1)

    norm2 = K.layers.BatchNormalization(axis=3)
    norm2 = norm2(conv_F3)

    X2 = K.layers.Activation('relu')(norm2)

    conv_F12 = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer)
    conv_F12 = conv_F12(X2)

    norm3 = K.layers.BatchNormalization(axis=3)
    norm3 = norm3(conv_F12)

    add = K.layers.Add()([norm3, A_prev])
    X3 = K.layers.Activation('relu')(add)

    return X3
