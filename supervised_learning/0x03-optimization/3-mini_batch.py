#!/usr/bin/env python3
""" contains the train_mini_batch module"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network using mini batch gradient descent
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        mbiter = X_train.shape[0]/batch_size

        if (mbiter).is_integer() is True:
            mbiter = int(mbiter)
        else:
            mbiter = (int(mbiter) + 1)

        for i in range(epochs + 1):
            tr_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            tr_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            val_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            val_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(tr_cost))
            print('\tTraining Accuracy: {}'.format(tr_acc))
            print('\tValidation Cost: {}'.format(val_cost))
            print('\tValidation Accuracy: {}'.format(val_acc))

            if i < epochs:
                X_s, Y_s = shuffle_data(X_train, Y_train)

                for j in range(mbiter):
                    fst = j * batch_size
                    lst = (j + 1) * batch_size
                    if lst > X_train.shape[0]:
                        lst = X_train.shape[0]
                    ndict = {x: X_s[fst:lst], y: Y_s[fst:lst]}
                    sess.run(train_op, feed_dict=ndict)
                    if j != 0 and (j + 1) % 100 == 0:
                        M_cost = sess.run(loss, feed_dict=ndict)
                        M_acc = sess.run(accuracy, feed_dict=ndict)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(M_cost))
                        print("\t\tAccuracy: {}".format(M_acc))

        save_path = saver.save(sess, save_path)
    return save_path
