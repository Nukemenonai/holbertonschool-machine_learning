#!/usr/bin/env python3
"""complete implementation with optimization techniques"""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data
train_mini_batch = __import__('3-mini_batch').train_mini_batch
create_Adam_op = __import__('10-Adam').create_Adam_op
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer


def forward_prop(x, layers, activations):
    """performs forward propagation on the neural network"""
    A = create_batch_norm_layer(x, layers[0], activations[0])
    for i in range(1, len(activations)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    return A


def calc_accuracy(y, y_pred):
    """computes accuracy of prediction"""
    y_idx = tf.math.argmax(y, axis=1)
    p_idx = tf.math.argmax(y_pred, axis=1)
    c = tf.math.equal(y_idx, p_idx)
    cast = tf.cast(c, dtype=tf.float32)
    acc = tf.math.reduce_mean(cast)
    return acc


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

    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]],
                       name='x')
    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]],
                       name='y')
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

        tr_inputs = {x: Data_train[0], y: Data_train[1]}
        val_inputs = {x: Data_valid[0], y: Data_valid[1]}

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
