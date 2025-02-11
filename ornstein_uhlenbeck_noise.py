"""
ornstein_uhlenbeck_noise.py
===========================

This file contains the OrnsteinUhlenbeckNoise class, which is a class that represents the Ornstein-Uhlenbeck noise
that is added to the actions in the DDPG algorithm. This type of noise introduces temporal correlation, making it
suitable for environments with momentum.

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    December 8, 2024
"""

import numpy


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, theta=0.15, sigma=0.2, dt=1e-2, mu=0.0, decay_rate=0.99, min_sigma=0.00):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.init_sigma = sigma
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma
        self.state = numpy.ones(self.action_dim) * self.mu

    def sample(self):
        noise = numpy.random.normal(size=self.action_dim)
        self.state += self.theta * (self.mu - self.state) * self.dt + self.sigma * numpy.sqrt(self.dt) * noise
        return self.state

    def decay(self):
        self.sigma = max(self.sigma * self.decay_rate, self.min_sigma)

    def reset(self):
        self.state = numpy.ones(self.action_dim) * self.mu
