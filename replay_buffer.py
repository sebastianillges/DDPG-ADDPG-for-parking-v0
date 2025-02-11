"""
replay_buffer.py
================

This file contains the ReplayBuffer class, which stores the experiences of the agent in a cyclic buffer. This class is
optimized for the parking-v0 environment from the Gymnasium package.

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    January 27, 2025
"""
import torch
import random
import numpy


class ReplayBuffer:
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.index = 0

    def add(
            self,
            state: dict[str, numpy.ndarray],
            action: torch.Tensor,
            reward: float,
            next_state: dict[str, numpy.ndarray],
            done: bool,
    ):
        """Add a new transition to the buffer."""
        # Detach tensors to avoid storing gradients
        state = {key: torch.tensor(value, dtype=torch.float32, device=self.device).clone().detach() for key, value in state.items()}
        next_state = {key: torch.tensor(value, dtype=torch.float32, device=self.device).clone().detach() for key, value in next_state.items()}
        action = action.detach()
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device).detach()
        done_tensor = torch.tensor([done], dtype=torch.float32, device=self.device).detach()

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward_tensor, next_state, done_tensor)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size: int):
        """Sample a batch of transitions from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack states and next_states
        states = {key: torch.stack([s[key] for s in states], dim=0).to(self.device) for key in states[0]}
        next_states = {key: torch.stack([ns[key] for ns in next_states], dim=0).to(self.device) for key in next_states[0]}

        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.stack(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)