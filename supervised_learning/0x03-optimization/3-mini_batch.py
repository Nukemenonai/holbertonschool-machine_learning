#!/usr/bin/env python3
""" contains the train_mini_batch module"""

import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains a neural network using mini batch gradient descent"""

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")[0]

        mbit = X_train.shape[0] / batch_size
        mbit = int(mbit + 1) if type(mbit) != int else int(bit)

        for i in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            val_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            val_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(val_cost))
            print("\tValidation Accuracy: {}".format(val_accuracy))

            if i < epochs:
                X_S, Y_S = shuffle_data(X_train, Y_train)
                for j in range(mbit):
                    first = j * batch_size
                    last = (j + 1) * batch_size
                    last = X_train.shape[0] if last > X_train.shape[0] else last
                    ndict = {x: X_S[first:last], y: Y_S[first:last]}
                    sess.run(train_op, feed_dict=ndict)
                    if j != 0 and (j + 1) % 100 == 0:
                        mcost = sess.run(loss, feed_dict=ndict)
                        maccuracy = sess.run(accuracy, feed_dict=ndict)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(mcost))
                        print("\t\tAccuracy: {}".format(maccuracy))

        saver.save(sess, save_path)
    return save_path
