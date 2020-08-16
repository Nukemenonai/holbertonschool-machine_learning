#!/usr/bin/env python3
"""
evaluates the output of the neural network
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of the NN"""
    saver = tf.train.import_meta_graph(save_path + '.meta')
    with tf.Session as sess:
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        output = ses.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = ses.run(accuracy, feed_dict={x: X, y: Y})
        cost = ses.run(loss, feed_dict={x: X, y: Y})
    return output, accuracy, cost
