"""
critic.py
========

This file contains the Critic class, which is a PyTorch neural network that represents the policy function of the DDPG
algorithm.

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    December 8, 2024
"""

import torch


class Critic(torch.nn.Module):
    def __init__(self, e, hidden_layers=[400, 300], activation_fn=torch.nn.ReLU, device=torch.device("cuda:0")):
        super().__init__()

        # The input dimension for the critic is the concatenation of the state and action
        input_dim = (
                e.observation_space['observation'].shape[0] +
                e.observation_space['achieved_goal'].shape[0] +
                e.observation_space['desired_goal'].shape[0] +
                e.action_space.shape[0]
        )
        output_dim = 1  # Critic outputs a single Q-value

        # Create the fully connected layers
        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim

        # Output layer (Q-value prediction)
        layers.append(torch.nn.Linear(last_dim, output_dim))

        # Define the network
        self.device = device
        self.q_value = torch.nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, observation_dict, action) -> torch.Tensor:
        observation_tensor = torch.cat([
            observation_dict["observation"],
            observation_dict["achieved_goal"],
            observation_dict["desired_goal"],
            action
        ], dim=-1)

        observation_tensor.to(self.device)

        # Call Q-value network
        return self.q_value(observation_tensor)
