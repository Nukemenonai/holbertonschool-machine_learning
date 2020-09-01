#!/usr/bin/env python3
"""trais a model using mini batch gradient descent"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    early_stopping boolean that indicates whether early stopping should be used
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    save_best is a boolean indicating whether to save the model
    after each epoch if it is the best
    a model is considered the best if its validation loss
    is the lowest that the model has obtained
    filepath where the model should be saved
    """
    callback_list = None

    if validation_data:
        callback_list = [K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)]
        if learning_rate_decay:
            def decayed_learning_rate(step):
                return alpha / (1 + decay_rate * step)
            c = K.callbacks.LearningRateScheduler(decayed_learning_rate,
                                                  verbose=1)
            callback_list.append(c)

        if save_best and filepath:
            s = callbacks.ModelCheckpoint(filepath,
                                          monitor='val_loss', verbose=0,
                                          save_best_only=False,
                                          save_weights_only=False,
                                          mode='min',
                                          save_freq='epoch',
                                          options=None)
            callback_list.append(s)

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=callback_list)
    return history
