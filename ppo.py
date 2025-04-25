import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gyms import LandingSpacecraftGym, LandingEvaluator, create_normalized_observation, mirror_observation
import os
from pathlib import Path
import optuna
import numpy as np


class LandingAgent:

    def __init__(self):
        self.ppo = PPO()
        self.ppo.load_state_dict(torch.load(
            Path.cwd() / "ppo" / "landing.pt", weights_only=True))
        self.action_space = LandingSpacecraftGym().discrete_action_space

    def get_action(self, env, target):
        obs = create_normalized_observation(env, target)
        return self.action_space[self.ppo.pi(torch.from_numpy(obs).float()).argmax()]


class TransitionBuffer:
    def __init__(self):
        self.data = []

    def create_batch(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        action_probabilities = []
        dones = []

        for transition in self.data:
            state, action, reward, next_state, action_prob, done = transition

            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            action_probabilities.append([action_prob])
            dones.append([1 if done else 0])

        return torch.tensor(np.array(states), dtype=torch.float), torch.tensor(np.array(actions), dtype=int), \
            torch.tensor(np.array(rewards)), torch.tensor(np.array(next_states), dtype=torch.float), \
            torch.tensor(np.array(dones), dtype=torch.float), torch.tensor(
                np.array(action_probabilities))

    def add_transition(self, transition):
        self.data.append(transition)

    def clear(self):
        self.data.clear()


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(9, 128),  # 9 observation inputs
            nn.ReLU(),
            nn.Linear(128, 9)  # 9 action
        )

    def pi(self, state):
        return self.policy_net(state).softmax(0 if len(state.shape) == 1 else 1)

    def value(self, state):
        return self.value_net(state)


def train_network(ppo: PPO,
                  transition_buffer: TransitionBuffer,
                  optimizer: torch.optim.Optimizer,
                  gamma,  # discount rate
                  lmbda,
                  eps_clip,  # clipping rate
                  K_epoch  # how many epochs to run for each batch
                  ):

    state, action, reward, next_state, dones, action_prob = transition_buffer.create_batch()

    for i in range(K_epoch):
        td_target = reward + gamma * ppo.value(next_state) * (1 - dones)
        delta = td_target - ppo.value(state)
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)

        pi = ppo.pi(state)
        pi_a = pi.gather(1, action)

        ratio = torch.exp(torch.log(pi_a) - torch.log(action_prob))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + \
            F.smooth_l1_loss(ppo.value(state), td_target.detach())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()


def train_landing_agent(learning_rate=0.0005,
                        n_episodes=4_000,
                        gamma=0.98,
                        lmbda=0.90,
                        eps_clip=0.2,
                        K_epoch=10,
                        T_horizon=600,
                        eval_freq=1,
                        verbose=True,
                        reward_scale=20,
                        # alpha to linearly decrease clipping
                        alpha_start=1,
                        alpha_end=0.25,
                        alpha_episodes=3_000
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
        # alpha = (n_episodes - episode)/n_episodes
        # alpha = alpha_start + episode * \
        #     (alpha_end - alpha_start)/alpha_episodes
        # alpha = max(alpha, alpha_end)
        alpha = 1
        state, _ = env.reset()
        episode_done = False
        while not episode_done:
            for t in range(T_horizon):
                prob = ppo.pi(torch.from_numpy(state).float())

                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, terminated, truncated, _ = env.step(
                    action)

                transition_buffer.add_transition(
                    (state, action, reward/reward_scale, next_state, prob[action].item(), terminated))

                state = next_state

                if terminated or truncated:
                    episode_done = True
                    break

            train_network(
                ppo,
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
                print("Alpha:", alpha)
                evaluator.print_results()

            evaluator.save_flight_trajectory_plot(
                training_directory / "trajectories" / f"{episode:05}.png")

            if avg_reward > highest_avg_reward:
                evaluator.save_flight_trajectory_plot(
                    training_directory / "best_flight_trajectories.png")
                highest_avg_reward = avg_reward
                torch.save(ppo.state_dict(),
                           training_directory / f"landing.pt")
            evaluator.append_to_csv(episode, eval_csv)
            print("Best:", highest_avg_reward)
    return highest_avg_reward


def test_mirrored_policy():
    ppo = PPO()
    ppo.load_state_dict(torch.load(
        Path.cwd() / "ppo" / "landing.pt", weights_only=True))

    action_space = LandingSpacecraftGym().discrete_action_space

    symmetry = "right"

    def get_action_fn(obs):
        # Use right side policy on the left side
        if symmetry == "right":
            if obs[0] < 0:
                obs = mirror_observation(obs)
                action = action_space[ppo.pi(torch.from_numpy(
                    obs).float()).detach().numpy().argmax()]
                mirrored_action = (action[0] * -1, action[1])
                for idx, a in enumerate(action_space):
                    if a == mirrored_action:
                        return idx
            else:
                return ppo.pi(torch.from_numpy(obs).float()).detach().numpy().argmax()

        # use left side policy on he right side
        elif symmetry == "left":
            if obs[0] > 0:
                obs = mirror_observation(obs)
                action = action_space[ppo.pi(torch.from_numpy(
                    obs).float()).detach().numpy().argmax()]
                mirrored_action = (action[0] * -1, action[1])
                for idx, a in enumerate(action_space):
                    if a == mirrored_action:
                        return idx
            else:
                return ppo.pi(torch.from_numpy(obs).float()).detach().numpy().argmax()
        else:
            return ppo.pi(torch.from_numpy(obs).float()).detach().numpy().argmax()

    evaluator = LandingEvaluator(discrete_actions=True)
    evaluator.evaluate(get_action_fn)

    print(evaluator.get_avg_reward())
    evaluator.save_flight_trajectory_plot(Path.cwd() / "ppo_mirrored_policy.png")


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
    # train_landing_agent()
    test_mirrored_policy()
