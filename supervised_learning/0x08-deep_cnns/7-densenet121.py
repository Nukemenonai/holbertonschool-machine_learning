#!/usr/bin/env python3
"""
Creating DenseNet
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds a densenet 121 architecture"""

    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()
    norm1 = K.layers.BatchNormalization(axis=3)(X)
    A = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=(2 * growth_rate),
                            kernel_size=(7, 7),
                            padding='same',
                            strides=(2, 2),
                            kernel_initializer=initializer)(A)
    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same')(conv1)
    dense1, filtr_d1 = dense_block(maxpool1, (2 * growth_rate), growth_rate, 6)
    transition1, filtr_t1 = transition_layer(dense1, filtr_d1, compression)
    dense2, filtr_d2 = dense_block(transition1, filtr_t1, growth_rate, 12)
    transition2, filtr_t2 = transition_layer(dense2, filtr_d2, compression)
    dense3, filtr_d3 = dense_block(transition2, filtr_t2, growth_rate, 24)
    transition3, filtr_t3 = transition_layer(dense3, filtr_d3, compression)
    dense4, filtr_d4 = dense_block(transition3, filtr_t3, growth_rate, 16)
    avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1))(dense4)
    FC = K.layers.Dense(units=1000, activation='softmax',
                        kernel_initializer=initializer)(avg)
    model = K.models.Model(inputs=X, outputs=FC)
    return model
