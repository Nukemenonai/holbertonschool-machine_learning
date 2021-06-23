#!/usr/bin/env python3
"""
Epsilon Greedy
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    uses epsilon-greedy to determine the next action:

    Q: numpy.ndarray containing the q-table
    state: current state
    epsilon: epsilon to use for the calculation
    Returns: the next action index
    """

    if np.random.uniform(0, 1) < epsilon:
        # explore
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit - choose optimal value 
        action = np.argmax(Q[state])
    return action
