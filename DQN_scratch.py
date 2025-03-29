import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gyms import SpacecraftGym, LandingSpacecraftGym
import gymnasium as gym
from spacecraft import Environment
from common import Settings
from collections import namedtuple, deque
from itertools import count
import random
import time

from torch.utils.tensorboard import SummaryWriter


Experience = namedtuple(
    'Experience', ('state', 'action', 'reward', 'next_state'))


class DQN:

    def __init__(self, env: SpacecraftGym):
        self.env = env
        self.discount_rate = 0.99
        self.memory = deque(maxlen=10_000)
        self.epsilon_start = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999
        self.epsilon = self.epsilon_start
        self.batch_size = 32
        self.tau = 0.5
        hidden_features = 64
        self.policy_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, env.action_space.n)
        )
        self.target_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, env.action_space.n)
        )
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=0.0001, amsgrad=True)

        self.criterion = torch.nn.SmoothL1Loss()

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return torch.argmax(self.policy_net(torch.tensor(state, dtype=torch.float)).detach()).item()
        else:
            return random.randint(0, self.env.action_space.n-1)

    def _convert_to_batch(self, experiences: list[Experience]):
        states = []
        actions = []
        rewards = []
        next_states = []
        not_final = []
        for experience in experiences:
            state, action, reward, next_state = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if next_state is not None:
                next_states.append(next_state)

            not_final.append(next_state is not None)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        not_final = torch.tensor(not_final)
        return states, actions, rewards, next_states, not_final

    def _fit_policy_network(self):
        if len(self.memory) < self.batch_size:
            return 0

        experiences = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, not_final = self._convert_to_batch(
            experiences)

        state_action_values = self.policy_net(states).gather(1, actions)

        next_state_action_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_action_values[not_final] = self.target_net(
                next_states).detach().max(1).values

        next_state_action_values = next_state_action_values
        expected_state_action_values = (
            rewards + next_state_action_values * self.discount_rate).unsqueeze(1)

        loss = self.criterion(state_action_values,
                              expected_state_action_values)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)

        self.optimizer.step()

        return loss.item()

    def _update_target_network(self):
        """
        Soft copy of policy network into target network
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.tau + target_net_state_dict[key]*(1-self.tau)

        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, n_episodes):
        run_name = "DQN_" + time.strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir="tensorboard_logs/" + run_name)

        step_count = 0
        episode_count = 0
        policy_network_update_count = 0

        for e in range(n_episodes * 1000):
            state, _ = self.env.reset()
            reward_sum = 0
            episode_step_count = 0
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_end

            for t in count():
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                episode_step_count += 1
                reward_sum += reward
                step_count += 1

                if terminated:
                    next_state = None

                experience = Experience(state, action, reward, next_state)

                self.memory.append(experience)

                loss = self._fit_policy_network()
                writer.add_scalar("Training/loss", loss,
                                  policy_network_update_count)
                policy_network_update_count += 1

                if step_count % 1000 == 0:
                    self._update_target_network()

                state = next_state

                if terminated or truncated:
                    episode_count += 1
                    break

            writer.add_scalar("Training/epsilon", self.epsilon, e)
            writer.add_scalar("Episode/reward", reward_sum, e)
            writer.add_scalar("Episode/steps", episode_step_count, e)
            writer.flush()


def selct_action(model, state, epsilon):
    if random.random() > epsilon:
        return torch.argmax(model(torch.tensor(state, dtype=torch.float)))
    else:
        return random.randint(0, 8)


if __name__ == "__main__":
    # Based on tutorial at https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    init_env = Environment(time_step_size=Settings.TIME_STEP_SIZE)
    gym_env = LandingSpacecraftGym(init_env)
    gym_env = gym.make("CartPole-v1")
    dqn = DQN(gym_env)
    dqn.train(100)
