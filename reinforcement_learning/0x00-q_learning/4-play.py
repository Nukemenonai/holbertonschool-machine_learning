#!/usr/bin/env python3
"""
Playing an episode
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    env: FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    max_steps: maximum number of steps in the episode
    Each state of the board is displayed via the console
    function always exploit the Q-table
    Returns: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    done = False
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        if done is True:
            env.render()
            return reward
        else:
            env.render()
            state = new_state