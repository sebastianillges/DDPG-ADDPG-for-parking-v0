"""
model.py
======

Super Class for DDPG and ADDPG

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    December 8, 2024
"""
import json
import os
import time
import torch

class Model(torch.nn.Module):

    def __init__(self, env):
        super().__init__()
        self.env = env

    def save(self, save_path: str, model_name: str, config: dict=None, results: dict=None):
        """Saves the actor, critic, and their target networks."""
        full_save_path = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", save_path, model_name))
        os.makedirs(full_save_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(full_save_path, "actor.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(full_save_path, "actor_target.pth"))
        torch.save(self.critic.state_dict(), os.path.join(full_save_path, "critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(full_save_path, "critic_target.pth"))
        with open(os.path.join(full_save_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        with open(os.path.join(full_save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"Model saved to {full_save_path}")

    def load(self, load_path: str, model_name: str):
        """Loads the actor, critic, and their target networks"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_save_path = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", load_path, model_name))
        self.actor.load_state_dict(torch.load(os.path.join(full_save_path, "actor.pth"), weights_only=True, map_location=device))
        self.actor_target.load_state_dict(
            torch.load(os.path.join(full_save_path, "actor_target.pth"), weights_only=True, map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(full_save_path, "critic.pth"), weights_only=True, map_location=device))
        self.critic_target.load_state_dict(
            torch.load(os.path.join(full_save_path, "critic_target.pth"), weights_only=True, map_location=device))
        print(f"Model loaded from {load_path}")
        print(f"Model loaded from {load_path}")

    def evaluate_model(self, env, n_episodes=5, render=False, timeout=15):
        """Evaluates a model on a given environment

        Args:
            env: gym environment object
            n_episodes (int, optional): number of episodes to evaluate the model on. Defaults to 5.
            render (bool, optional): render the environment. Defaults to False.
            timeout (int, optional): maximum time to evaluate the model. Defaults to 15.

        Returns:
            float: mean reward over n_episodes
        """

        def render_dummy():
            pass

        if not render:
            env.render = render_dummy
        tries = 0
        successes = 0
        obs, _ = env.reset()
        start_time = time.perf_counter()
        while tries < n_episodes and time.perf_counter() - start_time < timeout:
            action = self.predict(obs)
            obs, reward, done, truncated, info = env.step(action.detach().cpu().numpy())
            env.render()
            if done or truncated:
                tries += 1
                if info["is_success"]:
                    successes += 1
                obs, _ = env.reset()
        env.reset()
        if tries == 0:
            return 0
        return successes / tries
