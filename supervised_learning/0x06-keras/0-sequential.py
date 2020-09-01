#!/usr/bin/env python3
""" build_model module """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the kearas library"""
    L2 = K.regularizers.l2(lambtha)
    model = K.Sequential()
    for i in range(len(layers)):
        model.add(K.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=L2, input_shape=(nx,)))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
