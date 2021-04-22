#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    input_encoder = keras.layers.Input(shape=input_dims)
    input_encoded = input_encoder

    for i, n in enumerate(filters):
        encoded = keras.layers.Conv2D(n, (3, 3), activation='relu',
                                      padding='same')(input_encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
        input_encoded = encoded

    encoder = keras.models.Model(input_encoder, encoded)

    input_decoder = keras.layers.Input(shape=latent_dims)
    input_decoded = input_decoder

    for i, n in enumerate(filters[::-1]):
        if i == len(filters) - 1:
            decoded = keras.layers.Conv2D(n, (3, 3), activation='relu',
                                          padding='valid')(input_decoded)
        else:
            decoded = keras.layers.Conv2D(n, (3, 3), activation='relu',
                                          padding='same')(input_decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
        input_decoded = decoded

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                  activation='sigmoid',
                                  padding='same')(decoded)

    decoder = keras.models.Model(input_decoder, decoded)

    X_input = keras.layers.Input(shape=input_dims)
    encoder_out = encoder(X_input)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(X_input, decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto