import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gyms import LandingSpacecraftGym, LandingEvaluator, create_normalized_observation
import os
from pathlib import Path
import optuna
import numpy as np


class LandingAgent:

    def __init__(self):
        self.ppo = PPO()
        self.ppo.load(Path.cwd() / "ppo" / "landing.pt")
        self.action_space = LandingSpacecraftGym().discrete_action_space

    def get_action(self, env, target):
        obs = create_normalized_observation(env, target)
        return self.action_space[self.ppo.pi(torch.from_numpy(obs).float()).argmax()]


class TransitionBuffer:
    def __init__(self):
        self.data = []

    def create_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst), dtype=int), \
            torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
            torch.tensor(np.array(done_lst), dtype=torch.float), torch.tensor(
                np.array(prob_a_lst))
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def add_transition(self, transition):
        self.data.append(transition)

    def clear(self):
        self.data.clear()


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(9, 256)
        self.fc_pi = nn.Linear(256, 9)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def train_network(self,
                      transition_buffer: TransitionBuffer,
                      optimizer: torch.optim.Optimizer,
                      gamma,  # discount rate
                      lmbda,
                      eps_clip,  # clipping rate
                      K_epoch  # how many epochs to run for each batch
                      ):
        # Train on batch of transitions from the transition buffer
        s, a, r, s_prime, done_mask, prob_a = transition_buffer.create_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            # a/b == exp(log(a)-log(b))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + \
                F.smooth_l1_loss(self.v(s), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()


def train_landing_agent(learning_rate=0.0005,
                        n_episodes=3_000,
                        gamma=0.99,
                        lmbda=0.95,
                        eps_clip=0.1,
                        K_epoch=3,
                        T_horizon=200,
                        eval_freq=10,
                        verbose=True,
                        reward_scale=100,
                        ):

    env = LandingSpacecraftGym()
    evaluator = LandingEvaluator()
    transition_buffer = TransitionBuffer()
    ppo = PPO()
    optimizer = optim.Adam(ppo.parameters(), lr=learning_rate)
    highest_avg_reward = -float("inf")
    training_directory = Path.cwd() / "ppo"
    os.makedirs(training_directory, exist_ok=True)

    eval_csv = training_directory / f"eval.csv"

    for episode in range(n_episodes):
        alpha = (n_episodes - episode)/n_episodes
        s, _ = env.reset()
        episode_done = False
        while not episode_done:
            for t in range(T_horizon):
                prob = ppo.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)

                transition_buffer.add_transition(
                    (s, a, r/reward_scale, s_prime, prob[a].item(), terminated))

                s = s_prime

                if terminated or truncated:
                    episode_done = True
                    break

            ppo.train_network(
                transition_buffer=transition_buffer,
                optimizer=optimizer,
                gamma=gamma,
                lmbda=lmbda,
                eps_clip=eps_clip*alpha,
                K_epoch=K_epoch
            )
            transition_buffer.clear()

        if episode % eval_freq == 0 and episode > 0:
            evaluator.evaluate(lambda state: ppo.pi(
                torch.from_numpy(state).float()).argmax())

            avg_reward = evaluator.get_avg_reward()

            if verbose:
                print("Episode", episode)
                evaluator.print_results()

            evaluator.save_flight_trajectory_plot(
                training_directory / "latest_flight_trajectories.png")

            if avg_reward > highest_avg_reward:
                evaluator.save_flight_trajectory_plot(
                    training_directory / "best_flight_trajectories.png")
                highest_avg_reward = avg_reward
                torch.save(ppo.state_dict(),
                           training_directory / f"landing.pt")
            evaluator.append_to_csv(episode, eval_csv)
            print("Best:", highest_avg_reward)
    return highest_avg_reward


def optuna_objective(trial: optuna.Trial):

    return -train_landing_agent(
        n_episodes=2000,
        verbose=True,
        eval_freq=20,
        learning_rate=trial.suggest_float(
            "learning_rate", 0.0002, 0.001, step=0.0002),
        T_horizon=trial.suggest_int("T_horizon", 100, 1000, step=100),
        K_epoch=trial.suggest_int("K_epoch", 1, 5, step=1),
        eps_clip=trial.suggest_float("eps_clip", 0.1, 0.3, step=0.1),
        lmbda=trial.suggest_float("lambda", 0.9, 1, step=0.1),
        gamma=trial.suggest_float("gamma", 0.9, 1, step=0.01),
        reward_scale=trial.suggest_int("reward_scale", 5, 50, step=5)
    )


def find_good_hyperparams():
    storage = f"sqlite:///ppo/ppo_landing.db"
    study = optuna.create_study(study_name="ppo_landing", storage=storage)
    study.optimize(optuna_objective, n_trials=100)


if __name__ == '__main__':
    # find_good_hyperparams()
    train_landing_agent()
