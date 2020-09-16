#!/usr/bin/env python3
"""
inception block module
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """builds an inception block
    A_prev: previous layer output
    filters: contains F1 F3R F3 F5R F5 and FPP"""
    F1, F3R, F3, F5R, F5, FPP = filters
    # 1x1 convolution
    conv1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')
    conv1 = conv1(A_prev)
    # 1x1 conv before 3x3
    conv3R =K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')
    # 3x3 convolution
    conv3 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')
    conv3 = conv3(conv3R(A_prev))
    # 1x1 convolution before 5x5
    conv5R = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')
    # 5x5 convolution
    conv5 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')
    conv5 = conv5(conv5R(A_prev))
    # 3x3 max pooling
    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    # 1x1 conv after pooling
    FPP = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')
    FPP = FPP(pool(A_prev))

    # concatenation
    layer_out = K.layers.concatenate([conv1, conv3, conv5, FPP])
    return layer_out
