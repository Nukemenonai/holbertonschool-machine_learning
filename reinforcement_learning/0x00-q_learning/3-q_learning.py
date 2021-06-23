#!/usr/bin/env python3
"""
Q learning
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    env: FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: decay rate for updating epsilon between episodes
    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """

    total_rewards = []

    for i in range(episodes):
        state = env.reset()
        done = False
        # reward current episode
        reward_c_e = 0

        for _ in range(max_steps):
            # Exploration-explotation trade-off

            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            # Update Q-table for Q(s, a)
            part = (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            Q[state, action] += alpha * part
            state = new_state

            if done is True:
                if reward == 0.0:
                    # if agent falls in a hole
                    reward_c_e = -1
                reward_c_e += reward
                break

            reward_c_e += reward
        total_rewards.append(reward_c_e)
        part = (1 - min_epsilon) * np.exp(-epsilon_decay * i)
        epsilon = min_epsilon + part
    return Q, total_rewards