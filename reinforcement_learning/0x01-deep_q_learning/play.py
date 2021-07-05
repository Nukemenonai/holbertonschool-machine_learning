#!/usr/bin/env python3
"""
Agent training
"""

import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import keras as K
import numpy as np

create_q_model = __import__('train').q_learning_model
Atari2DProcessor = __import__('train').Atari2DProcessor


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    window = 4

    model = create_q_model(nb_actions, window)
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = Atari2DProcessor()

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   processor=processor,
                   memory=memory)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    # load weights.
    dqn.load_weights('policy.h5')

    # evaluate algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)
