import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def moving_average_smooth(values: np.ndarray, width=100):
    smoothed = np.zeros(values.shape)

    for i in range(smoothed.shape[0]):
        smoothed[i] = values[max(0, i - width//2):min(i+width//2, values.shape[0]-1)].mean()

    return smoothed


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
    idx = 3
    ppo_landings = ppo_csv[:, idx]
    ddpg_landings = ddpg_csv[:, idx]
    sac_landings = sac_csv[:, idx]

    plt.plot(eps, moving_average_smooth(ppo_landings), label="PPO")
    plt.plot(eps, moving_average_smooth(ddpg_landings), label="DDPG")
    plt.plot(eps, moving_average_smooth(sac_landings), label="SAC")

    plt.xlabel("Episode")
    plt.ylabel("# of successful landings")
    plt.legend()

    plt.show()


create_landing_plots()
