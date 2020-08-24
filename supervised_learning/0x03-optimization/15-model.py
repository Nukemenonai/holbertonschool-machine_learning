#!/usr/bin/env python3
"""complete implementation with optimization techniques"""

import numpy as np
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    We have to use this function only in the last layer
    because we dont have to normalize the output
    Returns: the tensor output of the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    A = tf.layers.Dense(units=n, name='layer', activation=activation,
                        kernel_initializer=init)
    Y_pred = A(prev)
    return (Y_pred)


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    xC = np.copy(X)
    yC = np.copy(Y)
    assert len(xC) == len(yC)
    r = np.random.permutation(len(xC))
    return xC[r], yC[r]

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creates Adam optimization operation with tensorflow"""
    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1,
                                  beta2=beta2,
                                  epsilon=epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates the learning rate decay operation using tensorflow"""
    return tf.train.inverse_time_decay(alpha,
                                       global_step,
                                       decay_step,
                                       decay_rate,
                                       staircase=True)


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network
    in tensorflow
    prev: activated output of previous layer
    n: number of nodes in layer"""
    if not activation:
        return create_layer(prev, n, activation)
 
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = layers(prev)

    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)

    epsilon = tf.constant(1e-8)

    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)

    return activation(Z_norm)

def forward_prop(x, layers, activations):
    """performs forward propagation on the neural network"""
    A = create_batch_norm_layer(x, layers[0], activations[0])
    for i in range(1, len(activations)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    return A    


def calc_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    acc = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(acc, tf.float32))


def calc_loss(y, y_pred):
    """computes the loss of the prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def model(Data_train, Data_valid, layers, activations, 
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, 
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """builds trains and saves a neural network model in tensorflow using 
    adam optimization, mini batch gradient descent,  learning rate decay and 
    batch optimization"""

    mbiter = Data_train[0].shape[0] / batch_size
    if (mbiter).is_integer() is True:
        mbiter = int(mbiter)
    else:
        mbiter = (int(mbiter) + 1)

    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]], name='y')
    y_pred = forward_prop(x, layers, activations)
    accuracy = calc_accuracy(y, y_pred)
    loss = calc_loss(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        tr_inputs = {x:Data_train[0], y:Data_train[1]}
        val_inputs = {x:Data_valid[0], y:Data_valid[1]}

        for i in range(epochs + 1):
            tr_cost = sess.run(loss, tr_inputs)
            tr_acc = sess.run(accuracy, tr_inputs)
            val_cost = sess.run(loss, val_inputs)
            val_acc = sess.run(accuracy, val_inputs)
            print("After {} epochs:".format(i))
            print('\tTraining Cost: {}'.format(tr_cost))
            print('\tTraining Accuracy: {}'.format(tr_acc))
            print('\tValidation Cost: {}'.format(val_cost))
            print('\tValidation Accuracy: {}'.format(val_acc))

            if i < epochs:
                X_s, Y_s = shuffle_data(Data_train[0], Data_train[1])
                sess.run(global_step.assign(i))
                sess.run(alpha)

                for j in range(mbiter):
                    fst = j * batch_size
                    lst = (j + 1) * batch_size
                    if lst > Data_train[0].shape[0]:
                        lst = Data_train[0].shape[0]
                    ndict = {x: X_s[fst:lst], y: Y_s[fst:lst]}
                    sess.run(train_op, feed_dict=ndict)
                    
                    if j != 0 and (j + 1) % 100 == 0:
                        mcost = sess.run(loss, ndict)
                        macc = sess.run(accuracy, ndict)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(mcost))
                        print("\t\tAccuracy: {}".format(macc))
        save_path = saver.save(sess, save_path)
    return save_path 
