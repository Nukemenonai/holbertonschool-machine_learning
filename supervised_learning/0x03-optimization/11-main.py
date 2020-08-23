#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    alpha_init = 0.1
    for i in range(100):
        alpha = learning_rate_decay(alpha_init, 1, i, 10)
        print(alpha)
