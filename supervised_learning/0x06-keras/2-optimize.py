#!/usr/bin/env python3
"""Adam optimization"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics:"""
    Adam = K.optimizers.Adam(learning_rate=alpha,
                             beta_1=beta1, beta_2=beta2)
    network.compile(loss=K.losses.CategoricalCrossentropy(),
                    optimizer=Adam, metrics=['accuracy'])
