#!/usr/bin/env python3
"""
Agent training
"""

import gym
import keras as K
from keras import layers

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from PIL import Image
import numpy as np


class Atari2DProcessor(Processor):
    """
    preprocessing
    """

    def process_observation(self, observation):
        """
        resizing and grayscale
        """
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)

        # resize and convert to grayscale
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')

        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)

        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Rescale
        We could perform this processing step in `process_observation`.
        In this case, however, we would need to store a `float32` array
        instead, which is 4x more memory intensive than an `uint8` array.
        This matters if we store 1M observations.
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
        rewards between -1 and 1
        """
        return np.clip(reward, -1., 1.)


def q_learning_model(num_actions, window):
    """
    model to use in the deep Q_learning process
    We use the same model that was described by
    Mnih et al. (2015).
    """
    # change sequencial model to input style
    input = layers.Input(shape=(window, 84, 84))
    process_input = layers.Permute((2, 3, 1))(input)

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu",
                           data_format="channels_last")(process_input)

    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu",
                           data_format="channels_last")(layer1)

    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu",
                           data_format="channels_last")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)

    layer6 = layers.Dense(64, activation="relu")(layer5)

    nb_actions = layers.Dense(num_actions, activation="linear")(layer6)

    model = K.Model(inputs=input, outputs=nb_actions)
    return model


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    nb_actions = env.action_space.n
    window = 4

    # deep convolutional neural network model
    model = q_learning_model(nb_actions, window)
    model.summary()

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
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])

    #callbacks

    callbacks = [ModelIntervalCheckpoint('policy.h5', interval=250000)]
    callbacks += [FileLogger('dqn_log.json', interval=100)]

    # training
    dqn.fit(env,
            callbacks=callbacks,
            nb_steps=1500000,
            log_interval=10000,
            visualize=True,
            verbose=2)

    # save the final weights.
    dqn.save_weights('policy.h5', overwrite=True)