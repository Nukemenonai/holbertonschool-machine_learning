#!/usr/bin/env python3
""" lenet 5 keras"""


import tensorflow.keras as K


def lenet5(X):
    """
    implementation of LeNet-5 with keras
    """
    init_layer = K.initializers.he_normal(seed=None)
    conv_layer1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                  padding='same', activation='relu',
                                  kernel_initializer=init_layer)
    conv1 = conv_layer1(X)
    pool_layer1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool1 = pool_layer1(conv1)
    conv_layer2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                  padding='valid', activation='relu',
                                  kernel_initializer=init_layer)
    conv2 = conv_layer2(pool1)
    pool_layer2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool2 = pool_layer2(conv2)
    x_vector = K.layers.Flatten()(pool2)
    layer3 = K.layers.Dense(units=120, activation='relu',
                            kernel_initializer=init_layer)
    FC3 = layer3(x_vector)
    layer4 = K.layers.Dense(units=84, activation='relu',
                            kernel_initializer=init_layer)
    FC4 = layer4(FC3)
    layer5 = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=init_layer)
    y_pred = layer5(FC4)
    model = K.models.Model(inputs=X, outputs=y_pred)
    Adam = K.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=Adam,
                  metrics=['accuracy'])
    return model
