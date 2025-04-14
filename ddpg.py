import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gyms import LandingSpacecraftGym, LandingEvaluator, create_normalized_observation
from pathlib import Path
import os
from common import ReplayBuffer


# Hyperparameters
lr_mu = 0.0005
lr_q = 0.001
gamma = 0.99
batch_size = 32
buffer_limit = 50000
tau = 0.005  # for target network soft update


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(9, 64)
        self.fc_a = nn.Linear(2, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta = 0.1
        self.dt = 0.01
        self.sigma = 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(
            param_target.data * (1.0 - tau) + param.data * tau)


class LandingAgent:

    def __init__(self):
        self.pi = MuNet()
        self.pi.load_state_dict(torch.load(
            Path.cwd() / "ddpg" / "landing.pt", weights_only=True))

    def get_action(self, env, target):
        obs = create_normalized_observation(env, target)
        return self.pi(torch.from_numpy(obs).float()).detach().numpy()


def main():
    env = LandingSpacecraftGym(discrete_actions=False)
    evaluator = LandingEvaluator(discrete_actions=False)
    training_directory = Path.cwd() / "ddpg"
    os.makedirs(training_directory, exist_ok=True)
    memory = ReplayBuffer(50000)
    reward_scaling = 10

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

    # "to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia" - https://arxiv.org/pdf/1509.02971
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(2))

    best_score = -float("inf")

    for episode in range(10000):
        s, _ = env.reset()
        done = False

        while not done:
            a = mu(torch.from_numpy(s).float())
            a = np.clip(a.detach().numpy() + ou_noise(), -1, 1)
            s_prime, r, done, truncated, info = env.step(a)
            memory.add_transition((s, a, r/reward_scaling, s_prime, done))
            s = s_prime
            if truncated:
                break

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target,
                      memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)

        if episode % 20 == 0:
            evaluator.evaluate(lambda s: mu(
                torch.from_numpy(s).float()).detach().numpy())
            print("Episode", episode)
            evaluator.print_results()
            evaluator.save_flight_trajectory_plot(
                training_directory / "latest_flight_trajectories.png")
            avg_reward = evaluator.get_avg_reward()
            if avg_reward > best_score:
                torch.save(mu.state_dict(), training_directory / "landing.pt")
                best_score = avg_reward
                evaluator.save_flight_trajectory_plot(
                    training_directory / "best_flight_trajectories.png")

    env.close()


if __name__ == '__main__':
    main()
