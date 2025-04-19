import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def moving_mean_sd(values: np.ndarray, width=50):
    mean = np.zeros(values.shape)
    std = np.zeros(values.shape)

    for i in range(mean.shape[0]):
        mean[i] = values[max(0, i - width//2):min(i+width //
                                                  2, values.shape[0]-1)].mean()
        std[i] = values[max(0, i - width//2):min(i+width //
                                                 2, values.shape[0]-1)].std()

    return mean, std


def open_eval_csv(path):
    contentz = []
    with open(path) as csv:
        for line in csv.readlines():
            ep, reward, length, landings = line.split(",")

            ep = int(ep)
            reward = float(reward)
            length = float(length)
            landings = int(landings)

            contentz.append((ep, reward, length, landings))

    return np.array(contentz)


def create_length_plots():
    pass


def create_reward_plots():
    pass


def create_landing_plots():
    ppo_csv = open_eval_csv(Path.cwd() / "ppo" / "eval.csv")
    ddpg_csv = open_eval_csv(Path.cwd() / "ddpg" / "eval.csv")
    sac_csv = open_eval_csv(Path.cwd() / "sac" / "eval.csv")

    eps = sac_csv[:, 0]
    idx = 1
    ppo_landings = ppo_csv[:, idx]
    ddpg_landings = ddpg_csv[:, idx]
    sac_landings = sac_csv[:, idx]

    ppo_mean, ppo_sd = moving_mean_sd(ppo_landings)
    ddpg_mean, ddpg_sd = moving_mean_sd(ddpg_landings)
    sac_mean, sac_sd = moving_mean_sd(sac_landings)

    plt.plot(eps, ppo_mean, label="PPO", color="blue")
    plt.fill_between(eps, ppo_mean - ppo_sd, ppo_mean +
                     ppo_sd, alpha=0.05, color="blue")

    plt.plot(eps, ddpg_mean, label="DDPG", color="green")
    plt.fill_between(eps, ddpg_mean - ddpg_sd, ddpg_mean +
                     ddpg_sd, alpha=0.05, color="green")

    plt.plot(eps, sac_mean, label="SAC", color="red")
    plt.fill_between(eps, sac_mean - sac_sd, sac_mean +
                     sac_sd, alpha=0.05, color="red")

    plt.xlabel("Episode")
    plt.ylabel("# of successful landings")
    plt.legend()

    plt.show()


create_landing_plots()
