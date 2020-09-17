#!/usr/bin/env python3
"""inception network"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds an inception network"""
    input = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            strides=(2, 2), padding='same',
                            activation='relu')
    conv1 = conv1(input)

    maxpool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
    maxpool1 = maxpool1(conv1)

    conv2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                            padding='same',
                            activation='relu')
    conv2 = conv2(maxpool1)

    conv3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                            padding='same',
                            activation='relu')
    conv3 = conv3(conv2)

    maxpool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
    maxpool2 = maxpool2(conv3)

    inception1 = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64])

    maxpool3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
    maxpool3 = maxpool3(inception2)

    inception3 = inception_block(maxpool3, [192, 96, 208, 16, 48, 64])
    inception4 = inception_block(inception3, [160, 112, 224, 24, 64, 64])
    inception5 = inception_block(inception4, [128, 128, 256, 24, 64, 64])
    inception6 = inception_block(inception5, [112, 144, 288, 32, 64, 64])
    inception7 = inception_block(inception6, [256, 160, 320, 32, 128, 128])

    maxpool4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
    maxpool4 = maxpool4(inception7)

    inception8 = inception_block(maxpool4, [256, 160, 320, 32, 128, 128])
    inception9 = inception_block(inception8, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))
    avg_pool = avg_pool(inception9)

    dropout = K.layers.Dropout(0.4)
    dropout = dropout(avg_pool)

    FC1 = K.layers.Dense(1000, activation='softmax')
    FC1 = FC1(dropout)

    model = K.models.Model(inputs=input, outputs=FC1)
    return model
