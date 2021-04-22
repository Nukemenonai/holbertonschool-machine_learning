#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """
    input_encoder = keras.layers.Input(shape=(input_dims,))
    input_encoded = input_encoder
    for n in hidden_layers:
        encoded = keras.layers.Dense(n, activation='relu')(input_encoded)
        input_encoded = encoded
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.models.Model(input_encoder, latent)

    input_decoder = keras.layers.Input(shape=(latent_dims,))
    input_decoded = input_decoder
    for i, n in enumerate(hidden_layers[::-1]):
        decoded = keras.layers.Dense(n, activation='relu')(input_decoded)
        input_decoded = decoded
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.models.Model(input_decoder, decoded)

    X_input = keras.layers.Input(shape=(input_dims,))
    encoder_out = encoder(X_input)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(X_input, decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
