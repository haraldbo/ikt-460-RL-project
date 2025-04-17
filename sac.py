import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from gyms import LandingSpacecraftGym, LandingEvaluator, create_normalized_observation
from pathlib import Path
import os
from common import ReplayBuffer
import optuna
import time


class PolicyTrainingNet(nn.Module):
    def __init__(self, learning_rate, init_alpha, lr_alpha, target_entropy):
        super(PolicyTrainingNet, self).__init__()
        self.target_entropy = target_entropy
        self.fc1 = nn.Linear(9, 128)
        self.fc_mu = nn.Linear(128, 2)
        self.fc_std = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - \
            torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() *
                       (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc_mu = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        return torch.tanh(mu)


class LandingAgent:

    def __init__(self):
        self.pi = PolicyNet()
        self.pi.load_state_dict(torch.load(
            Path.cwd() / "sac" / "landing.pt", weights_only=True))

    def get_action(self, env, target):
        obs = create_normalized_observation(env, target)
        return self.pi(torch.from_numpy(obs).float()).detach().numpy()


class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(9, 64)
        self.fc_a = nn.Linear(2, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target, tau):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - tau) + param.data * tau)


def calc_target(pi, q1, q2, mini_batch, gamma):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        # summing the log probs. It should be the same as multiplying together the probs?
        # Assuming thrust and gimbaling are indepent, but they mat not be.. Hmm
        entropy = -pi.log_alpha.exp() * log_prob.sum(1, keepdim=True)
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target


def train_landing_agent(
        n_episodes=5000,
        lr_pi=0.0005,
        lr_q=0.001,
        init_alpha=0.01,  # initial value of the entropy regularization coef
        gamma=0.98,  # discount factor
        batch_size=32,  # number of transitions to sample for each training loop
        n_batches=20,  # how many batches to train on after each episode
        buffer_size=50000,  # number of transitions to store int he replay buffer
        tau=0.01,  # for target network soft update
        target_entropy=-1.0,  # for automated alpha update
        lr_alpha=0.001,  # for automated alpha update
        reward_scale=15.0,
        eval_freq=10
):
    env = LandingSpacecraftGym(discrete_actions=False)
    evaluator = LandingEvaluator(discrete_actions=False)
    eval_net = PolicyNet()
    training_directory = Path.cwd() / "sac"

    os.makedirs(training_directory, exist_ok=True)
    eval_csv = training_directory / f"eval.csv"

    memory = ReplayBuffer(buffer_size=buffer_size)
    q1 = QNet(lr_q)
    q2 = QNet(lr_q)
    q1_target = QNet(lr_q)
    q2_target = QNet(lr_q)
    pi = PolicyTrainingNet(
        learning_rate=lr_pi,
        init_alpha=init_alpha,
        lr_alpha=lr_alpha,
        target_entropy=target_entropy
    )

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    highest_avg_reward = -float('inf')

    for episode in range(n_episodes):
        s, _ = env.reset()
        done = False

        while not done:
            a, log_prob = pi(torch.from_numpy(s).float())
            a = a.detach().numpy()
            s_prime, r, done, truncated, info = env.step(a)
            memory.add_transition((s, a, r/reward_scale, s_prime, done))
            s = s_prime
            if truncated:
                break

        if memory.size() > 1000:
            for i in range(n_batches):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(
                    pi, q1_target, q2_target, mini_batch, gamma)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target, tau)
                q2.soft_update(q2_target, tau)

        if episode % eval_freq == 0 and episode > 0:
            eval_net.fc1.load_state_dict(pi.fc1.state_dict())
            eval_net.fc_mu.load_state_dict(pi.fc_mu.state_dict())

            evaluator.evaluate(lambda s: eval_net(
                torch.from_numpy(s).float()).detach().numpy())
            print("Episode", episode)
            evaluator.print_results()
            evaluator.save_flight_trajectory_plot(
                training_directory / "latest_flight_trajectories.png")
            avg_reward = evaluator.get_avg_reward()
            if avg_reward > highest_avg_reward:
                torch.save(eval_net.state_dict(),
                           training_directory/"landing.pt")
                highest_avg_reward = avg_reward
                evaluator.save_flight_trajectory_plot(
                    training_directory / "best_flight_trajectories.png")
            print("Best:", highest_avg_reward)

            evaluator.append_to_csv(episode, eval_csv)

    return highest_avg_reward


def optuna_objective(trial: optuna.Trial):
    """

    lr_pi=0.0005,
        lr_q=0.001,
        init_alpha=0.01,
        gamma=0.98,
        batch_size=32,
        buffer_size=50000,
        tau=0.01,  # for target network soft update
        target_entropy=-1.0,  # for automated alpha update
        lr_alpha=0.001, 
    """

    """
    "We also examine how sensitive SAC is to some of
    the most important hyperparameters, namely reward scaling
    and target value update smoothing constant."
    """

    return -train_landing_agent(
        n_episodes=2000,
        lr_pi=trial.suggest_float("lr_pi", 0.0002, 0.001, step=0.0002),
        lr_q=trial.suggest_float("lr_q", 0.0005, 0.002, step=0.0005),
        init_alpha=0.01,  # initial value of the entropy regularization coef
        gamma=trial.suggest_float(
            "gamma", 0.90, 1, step=0.01),  # discount factor
        # number of transitions to sample for each training loop
        batch_size=trial.suggest_int("batch_size", 16, 128, step=16),
        # how many batches to train on after each episode
        n_batches=trial.suggest_int("n_batches", 5, 50, step=5),
        buffer_size=50000,  # number of transitions to store int he replay buffer
        # for target network soft update
        tau=trial.suggest_float("tau", 0.0001, 0.001, step=0.0001),
        target_entropy=-1.0,  # for automated alpha update
        lr_alpha=0.001,  # for automated alpha update
        reward_scale=trial.suggest_int("reward_scale", 2, 52, step=5)
    )


def find_good_hyperparams():
    storage = f"sqlite:///sac/sac_landing.db"
    study = optuna.create_study(study_name="sac_landing", storage=storage)
    study.optimize(optuna_objective, n_trials=200)


if __name__ == '__main__':
    train_landing_agent()
    # find_good_hyperparams()
