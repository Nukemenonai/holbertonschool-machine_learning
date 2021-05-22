#!/usr/bin/env python3
"""
Self Attention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Calculates attention for machne translation based on paper:
    NEURAL MACHINE TRANSLATION
    BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
    """

    def __init__(self, units):
        """
        * units: integer representing the number of hidden
          units in the alignment model
        * Sets the following public instance attributes:
        * W - a Dense layer with units units, to be applied to
          the previous decoder hidden state
        * U - a Dense layer with units units, to be applied to
          the encoder hidden states
        * V - a Dense layer with 1 units, to be applied to the
          tanh of the sum of the outputs of W and U
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        * s_prev: tensor of shape (batch, units) containing
          the previous decoder hidden state
        * hidden_states: tensor of shape (batch, input_seq_len, units)
          containing the outputs of the encoder
        Returns: context, weights
        * context: tensor of shape (batch, units) that contains the
          context vector for the decoder
        * weights: tensor of shape (batch, input_seq_len, 1) that
          contains the attention weights
        """
        s_expanded = tf.expand_dims(input=s_prev, axis=1)
        first = self.W(s_expanded)
        second = self.U(hidden_states)
        score = self.V(tf.nn.tanh(first + second))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights
