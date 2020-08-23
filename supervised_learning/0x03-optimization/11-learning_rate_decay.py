#!/usr/bin/env python3
"""learning rate decay module """


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates learning rate using inverse time decay
    alpha: learning rate
    decay_rate: weith used to determine rate at which alpha decays
    global_step: number of passes of GD elapsed
    decay_step: number of passes that should occur before alpha decays
    Return: updated alpha 
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))


