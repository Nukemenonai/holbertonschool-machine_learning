#!/usr/bin/env python3

import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """ that builds, trains, and saves a neural network classifier:"""
    x, y = create_placeholders(784, 10)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            tr_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            tr_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            val_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            val_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(tr_cost))
                print("\tTraining Accuracy: {}".format(tr_acc))
                print("\tValidation Cost: {}".format(val_cost))
                print("\tValidation Accuracy: {}".format(val_acc))
    return saver.save(sess, save_path)
