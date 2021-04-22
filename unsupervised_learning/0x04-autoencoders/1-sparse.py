#!/usr/bin/env python3
"""
Sparse Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoenncoder
    """
    reg = keras.regularizers.l1(lambtha)
    #  °°°°°°°°°°°°°°°°°°°°°° encoder °°°°°°°°°°°°°°°°°°°°°°°°°°
    X_encode = keras.Input(shape=(input_dims,))
    hidden_e = keras.layers.Dense(units=hidden_layers[0], activation='relu',
                                  activity_regularizer=reg)
    Y_prev = hidden_e(X_encode)
    for i in range(1, len(hidden_layers)):
        hidden_e = keras.layers.Dense(units=hidden_layers[i],
                                      activation='relu',
                                      activity_regularizer=reg)
        Y_prev = hidden_e(Y_prev)

    latent = keras.layers.Dense(units=latent_dims, activation='relu',
                                    activity_regularizer=reg)
    Y_encoded = latent(Y_prev)
    encoder = keras.Model(inputs=X_encode, outputs=Y_encoded)
    # encoder.summary()

    # °°°°°°°°°°°°°°°°°°°°°°°° decoder °°°°°°°°°°°°°°°°°°°°°°°°°°°
    X_decode = keras.Input(shape=(latent_dims,))
    hidden_d = keras.layers.Dense(units=hidden_layers[-1], activation='relu')
    Y_prev = hidden_d(X_decode)
    for j in range(len(hidden_layers) - 2, -1, -1):
        hidden_d = keras.layers.Dense(units=hidden_layers[j],
                                      activation='relu')
        Y_prev = hidden_d(Y_prev)
    last_layer = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_layer(Y_prev)
    decoder = keras.Model(inputs=X_decode, outputs=output)
    # decoder.summary()

    # °°°°°°°°°°°°°°°°°°°°°° model °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
    X_input = keras.Input(shape=(input_dims,))
    encode_out = encoder(X_input)
    decoder_out = decoder(encode_out)
    auto = keras.Model(inputs=X_input, outputs=decoder_out)
    # auto.summary()
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
