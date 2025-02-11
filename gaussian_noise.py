"""
gaussian_noise.py
=================

This file contains the GaussianNoise class, which is a class that represents the Gaussian noise that is added to the
actions in the DDPG algorithm.

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    December 8, 2024
"""

import numpy


class GaussianNoise:
    def __init__(self, action_dim, sigma=0.1, decay_rate=0.99, min_std=0.0):
        self.action_dim = action_dim
        self.init_std = sigma
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.min_std = min_std

    def sample(self):
        return numpy.random.normal(0.0, self.sigma, self.action_dim)

    def decay(self):
        self.sigma = max(self.sigma * self.decay_rate, self.min_std)

    def reset(self):
        pass
