#!/usr/bin/env python3
""" builds a model """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a NN with keras using input() """
    L2 = K.regularizers.l2(lambtha)
    X = K.Input(shape=(nx,))
    layer_l2 = K.layers.Dense(units=layers[0],
                              activation=activations[0], kernel_regularizer=L2)
    output = layer_l2(X)
    for i in range(1, len(layers)):
        dropout = K.layers.Dropout(keep_prob)
        Y = dropout(output)
        layer_l2 = K.layers.Dense(units=layers[i], activation=activations[i],
                            kernel_regularizer=L2)
        output = layer_l2(Y)
    Y_pred = output
    model = K.Model(inputs=X, outputs=Y_pred)
    return model
