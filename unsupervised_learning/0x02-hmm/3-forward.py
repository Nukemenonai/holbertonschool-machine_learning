#!/usr/bin/env python3
"""
The Forward Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model:

        Observation: (T,)
        contains the index of the observation
        T: number of observations
        Emission: (N, M),  the emission
        probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the hidden state i
        N: number of hidden states
        M: number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N),
        the transition probabilities
        Transition[i, j] is the probability of transitioning
        from the hidden state i to j
        Initial: (N, 1)  the probability of starting in a particular hidden state
        Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F: (N, T) the forward path probabilities
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            a = (F[:, t - 1] * Transition[:, s]) * \
                   Emission[s, Observation[t]]
            F[s, t] = np.sum(a)
    P = np.sum(F[:, -1])

    return P, F