#!/usr/bin/env python3
"""creates the forward propagation graph for the NN"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
        """ """
        A = create_layer(x, layer_sizes[0], activations[0])
        for i in range(1, len(layer_sizes)):
            Y = create_layer(A, layer_sizes[i], activations[i])
        return Y
