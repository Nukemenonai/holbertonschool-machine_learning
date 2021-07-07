#!/usr/bin/env python3
"""
Agent plays
"""

import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

import keras as K
import numpy as np

create_q_model = __import__('train').q_learning_model
Atari2DProcessor = __import__('train').Atari2DProcessor


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    np.random.seed(42)
    env.seed(42)
    nb_actions = env.action_space.n
    window = 4

    model = create_q_model(nb_actions, window)
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = Atari2DProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000000)


    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   nb_steps_warmup=10,
                   processor=processor,
                   memory=memory,
                   policy=policy)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # load weights.
    dqn.load_weights('policy.h5')

    # evaluate algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)
