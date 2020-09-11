#!/usr/bin/env python3
"""lenet 5 tensorflow"""


import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of LeNet-5 with tensorflow
    """

    init_layer = tf.contrib.layers.variance_scaling_initializer()

    activation = tf.nn.relu
    conv_layer1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                   padding='same', activation=activation,
                                   kernel_initializer=init_layer)
    conv1 = conv_layer1(x)

    pool_layer1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool1 = pool_layer1(conv1)

    conv_layer2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                   padding='valid', activation=activation,
                                   kernel_initializer=init_layer)
    conv2 = conv_layer2(pool1)

    pool_layer2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool2 = pool_layer2(conv2)

    x_vector = tf.layers.Flatten()(pool2)

    layer3 = tf.layers.Dense(units=120, activation=activation,
                             kernel_initializer=init_layer)
    FC3 = layer3(x_vector)

    layer4 = tf.layers.Dense(units=84, activation=activation
                             kernel_initializer=init_layer)
    FC4 = layer4(FC3)

    layer5 = tf.layers.Dense(units=10, kernel_initializer=init_layer)
    y_pred = layer5(FC4)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    idx_y = tf.math.argmax(y, axis=1)
    idx_pred = tf.math.argmax(y_pred, axis=1)
    c = tf.math.equal(idx_y, idx_pred)
    cast = tf.cast(c, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)

    Adam = tf.train.AdamOptimizer().minimize(loss)

    y_softmax = tf.nn.softmax(y_pred)
    return y_softmax, Adam, loss, accuracy
