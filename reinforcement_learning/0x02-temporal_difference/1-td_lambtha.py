#!/usr/bin/env python3
"""
TD(λ)
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Performs TD(λ) algorithm

    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate
    """
    for i in range(episodes):
        e = np.zeros(env.observation_space.n)
        state = env.reset()
        for j in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)

            delta = reward + gamma * V[new_state] - V[state]
            e[state] += 1.0

            V[state] = V[state] + alpha * delta * e[state]
            e[state] *= lambtha * gamma

            if done:
                break
            else:
                state = new_state

    return V