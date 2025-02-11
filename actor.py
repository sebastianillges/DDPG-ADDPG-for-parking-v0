"""
actor.py
========

This file contains the Actor class, which is a PyTorch neural network that represents the policy function of the DDPG
algorithm.

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    December 8, 2024
"""

import numpy
import torch


class Actor(torch.nn.Module):  # inherits from torch.nn.Module to use .to(device)
    def __init__(self, env, hidden_layers=[400, 300], activation_fn=torch.nn.ReLU, device=torch.device("cuda:0")):
        super().__init__()

        input_dim = (
                env.observation_space['observation'].shape[0] +
                env.observation_space['achieved_goal'].shape[0] +
                env.observation_space['desired_goal'].shape[0]
        )
        output_dim = env.action_space.shape[0]

        # Create the fully connected layers
        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim

        # Output layer
        layers.append(torch.nn.Linear(last_dim, output_dim))
        layers.append(torch.nn.Tanh())  # Squash output to [-1, 1]

        self.device = device
        self.mu = torch.nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, observation_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        observation_tensor = torch.cat([
            observation_dict["observation"],
            observation_dict["achieved_goal"],
            observation_dict["desired_goal"]
        ], dim=-1)

        # Call policy to get action
        return self.mu(observation_tensor)

    def predict(self, observation_dict: dict[str, numpy.ndarray], noise: numpy.ndarray = None) -> torch.Tensor:
        observation_tensor = {key: torch.tensor(observation_dict[key], dtype=torch.float32, device=self.device).clone().detach() for key in observation_dict}
        action = self.forward(observation_tensor)

        # Add noise if provided
        if noise is not None:
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=action.device).clone()
            action = action + noise_tensor
        return torch.clamp(action, -1, 1)
