#!/usr/bin/env python3
"""
Viterbi Algorithm
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    markov model
    Observation: is a numpy.ndarray of shape (T,) that contains
    the index of the observation
        T is the number of observations
    Emission: is a numpy.ndarray of shape (N, M) containing the
    emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given
        the hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition: is a 2D numpy.ndarray of shape (N, N) containing
    the transition probabilities
        Transition[i, j] is the probability of transitioning from
        the hidden state i to j
    Initial: a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    Return: path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of
        hidden states
        P is the probability of obtaining the path sequence
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    # backpointer
    bp = np.zeros((N, T))

    viterbi = np.zeros((N, T))
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            a = (viterbi[:, t - 1] * Transition[:, s]) * \
                   Emission[s, Observation[t]]
            viterbi[s, t] = np.max(a)

            bp[s, t] = np.argmax((viterbi[:, t - 1] * Transition[:, s]) *
                                 Emission[s, Observation[t]])
    P = np.max(viterbi[:, -1])

    S = np.argmax(viterbi[:, -1])

    path = [S]

    for t in range(T - 1, 0, -1):
        S = int(bp[S, t])
        path.append(S)

    return path[::-1], P
