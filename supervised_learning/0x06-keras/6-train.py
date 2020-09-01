#!/usr/bin/env python3
"""trais a model using mini batch gradient descent"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    """ trains a model using mini batch gradient descent
    validation_data is the data to validate the model with, if not None
    early_stopping boolean that indicates whether early stopping should be used
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    """
    callback_list = []

    if validation_data:
        callbacks_list.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience))
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=callbacks_list)
    return history
