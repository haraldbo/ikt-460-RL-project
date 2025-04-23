import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def moving_mean_sd(values: np.ndarray, width=200):
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
    ppo = ppo_csv[:, idx]
    ddpg = ddpg_csv[:, idx]
    sac = sac_csv[:, idx]

    ppo_mean, ppo_sd = moving_mean_sd(ppo)
    ddpg_mean, ddpg_sd = moving_mean_sd(ddpg)
    sac_mean, sac_sd = moving_mean_sd(sac)

    plt.plot(eps, ppo_mean, label="PPO", color="blue")
    plt.plot(eps, ppo, color="blue", alpha=0.3)
    ppo_max = np.argmax(ppo)
    plt.scatter([ppo_max], [ppo[ppo_max]], marker="*", color="blue")

    plt.plot(eps, ddpg_mean, label="DDPG", color="green")
    plt.plot(eps, ddpg, color="green", alpha=0.3)
    ddpg_max = np.argmax(ddpg)
    plt.scatter([ddpg_max], [ddpg[ddpg_max]], marker="*", color="green")

    plt.plot(eps, sac_mean, label="SAC", color="red")
    plt.plot(eps, sac, color="red", alpha=0.3)
    sac_max = np.argmax(sac)
    plt.scatter([sac_max], [sac[sac_max]], marker="*", color="red")

    plt.xlabel("Episode")
    plt.ylabel("# of successful landings")
    plt.legend()

    plt.show()


create_landing_plots()
