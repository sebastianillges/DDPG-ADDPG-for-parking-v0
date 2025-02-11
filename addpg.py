"""
addpg.py
=======

This script is a parallelized version of the DDPG algorithm. It is designed to work with the `parking-v0` environment
from the `gymnasium` package.

Author:
    Sebastian Illges (sebastian.illges@htwg-konstanz.de)

Date Created:
    December 5, 2024
"""
import datetime
import math
import time
import highway_env
import gymnasium
import torch

from model import Model
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from gaussian_noise import GaussianNoise
from ornstein_uhlenbeck_noise import OrnsteinUhlenbeckNoise


class ADDPG(Model):

    def __init__(self,
                 env,
                 learning_rate=0.001,
                 buffer_size=1_000_000,
                 batch_size=256,
                 tau=0.005,
                 gamma=0.99,
                 noise_type=OrnsteinUhlenbeckNoise,
                 action_noise_sigma=0.2,
                 noise_decay_rate=0.995,
                 eval_interval=25,
                 eval_episodes=100,
                 eval_timeout=5,
                 eval_goal=0.99,
                 device="cuda",
                 save_path="trained_models",
                 name=None,
                 number_of_lerners=8,
                 ):
        super().__init__(env)
        print("[INIT] Initializing ADDPG...")
        if device == "cuda":
            print("WARNING: CUDA device selected. This may cause issues with multiprocessing.")
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.action_noise_sigma = action_noise_sigma
        self.noise_decay_rate = noise_decay_rate
        self.noise_type = noise_type
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_timeout = eval_timeout
        self.eval_goal = eval_goal
        self.device = torch.device(device)
        self.to(device)
        self.number_of_lerners = number_of_lerners

        self.save_path = save_path
        self.name = name if name is not None else f"{self.__str__()}_{self.env.unwrapped.spec.id}_{datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")}"
        self.results = {}

        self.replay_buffer = ReplayBuffer(buffer_size, device=self.device)

        self.actor = Actor(self.env, device=self.device)
        self.actor.share_memory()
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.actor_target = Actor(self.env, device=self.device)
        self.actor_target.share_memory()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.env, device=self.device)
        self.critic.share_memory()
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_target = Critic(self.env, device=self.device)
        self.critic_target.share_memory()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.lock = torch.multiprocessing.Lock()
        self.result_queue = torch.multiprocessing.Queue()

        print(self.get_config())
        print("[INIT] ADDPG initialization complete.")

    def predict(self, observation_dict: dict) -> torch.Tensor:
        return self.actor.predict(observation_dict)

    def learn(self, max_time: int = math.inf):
        print(f"[LEARN] Starting learning with {self.number_of_lerners} learners...")
        start_event = torch.multiprocessing.Event()
        stop_event = torch.multiprocessing.Event()
        processes = []
        for i in range(self.number_of_lerners):
            p = torch.multiprocessing.Process(target=self._learn, args=(max_time, i, start_event, stop_event))
            processes.append(p)
            print(f"[LEARN] Starting learner process {i + 1}...")
            p.start()
        start_event.set()
        start_time = time.perf_counter()
        while not stop_event.is_set():
            time.sleep(self.eval_interval)
            print(f"[LEARN] Starting evaluation...")
            success_rate = self.evaluate_model(env=self.env, n_episodes=self.eval_episodes, timeout=self.eval_goal)
            print(f"[LEARN] Evaluation result: {success_rate:.2f}.")
            self.result_queue.put({"evaluation": {time.perf_counter() - start_time: success_rate}})
            if success_rate >= self.eval_goal:
                print(f"[LEARN] Success rate goal reached. Stopping training...")
                stop_event.set()
                break
        elapsed_time = time.perf_counter() - start_time
        for p in processes:
            print(f"[LEARN] Waiting for process {p} to finish...")
            p.join(timeout=1)
        print(f"\033[92m[LEARN] All processes finished after {elapsed_time}s.\033[0m")

        # Merge results of all workers
        while not self.result_queue.empty():
            results = self.result_queue.get()
            for key in results:
                if key in self.results:
                    self.results[key].update(results[key])
                else:
                    self.results[key] = results[key]
        # Sort
        for key in self.results:
            self.results[key] = dict(sorted(self.results[key].items()))

        self.save(save_path=self.save_path, model_name=self.name, config=self.get_config(), results=self.results)
        print(f"[LEARN] Saved model to {self.save_path}/{self.name}.")
        return self.results

    def _learn(self, max_time: int, idx=None, start_event=None, stop_event=None):
        env = gymnasium.make('parking-v0', render_mode='rgb_array')
        print(f"[PROCESS-{idx}] Waiting for start event...")
        start_event.wait()
        timesteps = 0
        episode = 0
        episode_return = 0
        episode_returns = {}
        episode_length = 0
        episode_lengths = {}

        state, _ = env.reset()
        action_noise = self.noise_type(self.env.action_space.shape[0], sigma=self.action_noise_sigma,
                                       decay_rate=self.noise_decay_rate)
        start_time = time.perf_counter()
        print(f"[PROCESS-{idx}] Starting learning at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}...")

        while time.perf_counter() - start_time < max_time:
            if stop_event.is_set():
                break
            action = self.actor.predict(state, noise=action_noise.sample())

            # Step the environment
            next_state, reward, done, truncated, info = env.step(action.detach().cpu().numpy())

            # Add to replay buffer
            with self.lock:
                self.replay_buffer.add(state, action, reward, next_state, done or truncated)

            # Update `state` for the next step
            state = next_state
            episode_return += reward
            episode_length += 1
            timesteps += 1

            if self.replay_buffer.size() > self.batch_size:
                self.train()

            if done or truncated:
                time_stamp = time.perf_counter() - start_time
                print(f"[Process-{idx} | {time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))}/{time.strftime("%H:%M:%S", time.gmtime(max_time))} | {timesteps}]: Episode: {episode}, Episode Return: {episode_return:.2f}, Episode length: {episode_length}")
                episode_returns[time_stamp] = episode_return
                episode_lengths[time_stamp] = episode_length
                state, _ = env.reset()

                episode += 1
                episode_return = 0
                episode_length = 0
                action_noise.decay()
                action_noise.reset()

        elapsed_time = time.perf_counter() - start_time
        env.close()
        print(f"[PROCESS-{idx}] Finished after {elapsed_time}s.")

        self.result_queue.put({"reward": episode_returns, "episode_lengths": episode_lengths})
        print(f"[PROCESS-{idx}] Saved training data....")
        stop_event.set()

    def train(self):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Update Critic
        q_targets = rewards + self.gamma * self.critic_target(next_states, self.actor_target(next_states)) * (1 - dones)
        q_values = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q_values, q_targets)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

    def __str__(self):
        return f"ADDPG{self.number_of_lerners}"

    def get_config(self):
        return {
            "model": self.__str__(),
            "env": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "noise_type": self.noise_type.__name__,
            "action_noise_sigma": self.action_noise_sigma,
            "noise_decay_rate": self.noise_decay_rate,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "eval_timeout": self.eval_timeout,
            "eval_goal": self.eval_goal,
            "device": self.device.type,
            "save_path": self.save_path,
            "name": self.name,
            "number_of_lerners": self.number_of_lerners
        }


if __name__ == '__main__':
    env = gymnasium.make('parking-v0', render_mode='rgb_array')
    addpg = ADDPG(env=env,
                  learning_rate=0.001,
                  buffer_size=1_000_000,
                  batch_size=256,
                  tau=0.005,
                  gamma=0.99,
                  noise_type=OrnsteinUhlenbeckNoise,
                  action_noise_sigma=0.2,
                  noise_decay_rate=0.995,
                  eval_interval=25,
                  eval_episodes=100,
                  eval_timeout=5,
                  eval_goal=0.99,
                  device="cpu",
                  save_path="trained_models",
                  name="ADDPG_test",
                  number_of_lerners=8
                  )
    addpg.learn(max_time=30)
    # addpg.load(load_path="trained_models", model_name="ADDPG_test")
    addpg.evaluate_model(env=env, n_episodes=100, render=True, timeout=60)
