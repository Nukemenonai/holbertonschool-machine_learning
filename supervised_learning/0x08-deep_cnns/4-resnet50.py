#!/usr/bin/env python3
"""
resnet50 as described on
Deep residual learning for image recognition (2015)
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds the ResNet-50 architecture"""
    input_layer = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            padding='same', strides=(2, 2),
                            kernel_initializer=initializer)(input_layer)

    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    X = K.layers.Activation('relu')(norm1)

    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding="same")(X)

    projection0 = projection_block(maxpool1, [64, 64, 256], 1)
    identity1 = identity_block(projection0, [64, 64, 256])
    identity2 = identity_block(identity1, [64, 64, 256])

    projection1 = projection_block(identity2, [128, 128, 512])

    identity3 = identity_block(projection1, [128, 128, 512])
    identity4 = identity_block(identity3, [128, 128, 512])
    identity5 = identity_block(identity4, [128, 128, 512])


    projection2 = projection_block(identity5, [256, 256, 1024])
    identity6 = identity_block(projection2, [256, 256, 1024])
    identity7 = identity_block(identity6, [256, 256, 1024])
    identity8 = identity_block(identity7, [256, 256, 1024])
    identity9 = identity_block(identity8, [256, 256, 1024])
    identity10 = identity_block(identity9, [256, 256, 1024])

    projection3 = projection_block(identity10, [512, 512, 2048])
    identity11 = identity_block(projection3, [512, 512, 2048])
    identity12 = identity_block(identity11, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1))(identity12)

    FC = K.layers.Dense(units=1000, activation='softmax',
                        kernel_initializer=initializer)(avg_pool)

    model = K.models.Model(inputs=input_layer, outputs=FC)
    return model
