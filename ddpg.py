import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gyms import LandingSpacecraftGym, LandingEvaluator, create_normalized_observation
from pathlib import Path
import os
from common import ReplayBuffer
import optuna


class LandingAgent:

    def __init__(self):
        self.pi = MuNet()
        self.pi.load_state_dict(torch.load(
            Path.cwd() / "ddpg" / "landing.pt", weights_only=True))

    def get_action(self, env, target):
        obs = create_normalized_observation(env, target)
        return self.pi(torch.from_numpy(obs).float()).detach().numpy()


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(11,  256),  # 9 state vars and 2 action vars
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


def train(mu, mu_target, q, q_target, memory: ReplayBuffer, q_optimizer, mu_optimizer, batch_size, gamma):
    state, action, reward, next_state, dones = memory.create_batch(batch_size)

    target = reward + gamma * \
        q_target(next_state, mu_target(next_state)) * dones
    q_loss = F.smooth_l1_loss(q(state, action), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(state, mu(state)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(
            param_target.data * (1.0 - tau) + param.data * tau)


def train_landing_agent(
        lr_mu=0.0006,
        lr_q=0.0008,
        gamma=0.99,
        batch_size=96,
        buffer_limit=1_000_000,
        tau=0.005,
        reward_scale=12,
        n_episodes=4_000,
        eval_freq=1,
        n_batches=50
):
    env = LandingSpacecraftGym(discrete_actions=False)
    evaluator = LandingEvaluator(discrete_actions=False)
    training_directory = Path.cwd() / "ddpg"
    eval_csv = training_directory / f"eval.csv"
    os.makedirs(training_directory, exist_ok=True)
    memory = ReplayBuffer(buffer_limit)

    q = QNet()
    q_target = QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

    highest_avg_reward = -float("inf")

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = mu(torch.from_numpy(state).float())
            action = np.clip(action.detach().numpy() +
                             np.random.normal(loc=0, size=2, scale=0.1), -1, 1)
            next_state, r, done, truncated, _ = env.step(action)
            memory.add_transition(
                (state, action, r/reward_scale, next_state, done))
            state = next_state
            if truncated:
                break

        if memory.size() > 1000:
            for i in range(n_batches):
                train(mu, mu_target, q, q_target,
                      memory, q_optimizer, mu_optimizer,
                      batch_size, gamma)
                soft_update(mu, mu_target, tau)
                soft_update(q,  q_target, tau)

        if episode % eval_freq == 0 and episode > 0:
            evaluator.evaluate(lambda s: mu(
                torch.from_numpy(s).float()).detach().numpy())
            print("Episode", episode)
            evaluator.print_results()
            evaluator.save_flight_trajectory_plot(
                training_directory / "trajectories" / f"{episode:05}.png")
            avg_reward = evaluator.get_avg_reward()
            if avg_reward > highest_avg_reward:
                torch.save(mu.state_dict(), training_directory / "landing.pt")
                highest_avg_reward = avg_reward
                evaluator.save_flight_trajectory_plot(
                    training_directory / "best_flight_trajectories.png")
            evaluator.append_to_csv(episode, eval_csv)
            print("Best:", highest_avg_reward)

    return highest_avg_reward


def optuna_objective(trial: optuna.Trial):

    return -train_landing_agent(
        lr_mu=trial.suggest_float("lr_mu", 0.0002, 0.001, step=0.0002),
        lr_q=trial.suggest_float("lr_q", 0.0002, 0.001, step=0.0002),
        gamma=trial.suggest_float("gamma", 0.9, 1, step=0.01),
        batch_size=trial.suggest_int("batch_size", 32, 128, step=32),
        tau=trial.suggest_float("tau", 0.0005, 0.005, step=0.0005),
        reward_scale=trial.suggest_int("reward_scale", 5, 50, step=5)
    )


def find_good_hyperparams():
    storage = f"sqlite:///ddpg/ddpg_landing.db"
    study = optuna.create_study(study_name="ddpg_landing", storage=storage)
    study.optimize(optuna_objective, n_trials=100)
    study.trials_dataframe().to_csv("ddpg_landing.csv")


if __name__ == '__main__':
    train_landing_agent()
    # find_good_hyperparams()
