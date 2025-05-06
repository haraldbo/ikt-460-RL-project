import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm


def moving_mean_sd(values: np.ndarray, width=100):
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


def create_landing_plots():
    ppo_csv = open_eval_csv(Path.cwd() / "report" / "ppo" / "eval.csv")
    ddpg_csv = open_eval_csv(Path.cwd() / "report" / "ddpg" / "eval.csv")
    sac_csv = open_eval_csv(Path.cwd() / "report" / "sac" / "eval.csv")

    n_episodes = 3999

    eps = sac_csv[:n_episodes, 0]

    show_max = True
    show_actual = True
    show_mean = True
    marker = "*"
    alpha = 0.3

    # reward = 1, length = 2, landings = 3
    stat_idx = 1

    ppo = ppo_csv[:n_episodes, stat_idx]
    ddpg = ddpg_csv[:n_episodes, stat_idx]
    sac = sac_csv[:n_episodes, stat_idx]

    ppo_mean, ppo_sd = moving_mean_sd(ppo)
    ddpg_mean, ddpg_sd = moving_mean_sd(ddpg)
    sac_mean, sac_sd = moving_mean_sd(sac)
    plt.figure(dpi=300)

    if show_mean:
        plt.plot(eps, ppo_mean, label="PPO", color="blue")
    if show_actual:
        plt.plot(eps, ppo, color="blue", alpha=alpha)
    if show_max:
        ppo_max = np.argmax(ppo)
        print("ppo max", ppo_max, ppo[ppo_max], ppo_csv[ppo_max, 2])
        plt.scatter([ppo_max], [ppo[ppo_max]], marker=marker, color="blue")

    if show_mean:
        plt.plot(eps, ddpg_mean, label="DDPG", color="green")
    if show_actual:
        plt.plot(eps, ddpg, color="green", alpha=alpha)
    if show_max:
        ddpg_max = np.argmax(ddpg)
        print("ddpg max", ddpg_max, ddpg[ddpg_max], ddpg_csv[ddpg_max, 2])
        plt.scatter([ddpg_max], [ddpg[ddpg_max]], marker=marker, color="green")

    if show_mean:
        plt.plot(eps, sac_mean, label="SAC", color="red")
    if show_actual:
        plt.plot(eps, sac, color="red", alpha=alpha)
    if show_max:
        sac_max = np.argmax(sac)
        print("SAC max", sac_max, sac[sac_max], sac_csv[sac_max, 2])
        plt.scatter([sac_max], [sac[sac_max]], marker=marker, color="red")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # plt.ylabel("Reward")
    # plt.ylabel("# of successful landings")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("rewards.png")


def create_evolution_movie(trajectory_directory):
    width = 640
    height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    video = cv2.VideoWriter('video.avi', fourcc, fps, (width, height))
    n_frames = 1000
    font = ImageFont.load_default(24)

    for i, file_name in enumerate(tqdm(os.listdir(trajectory_directory), desc="Creating demo")):
        img = Image.open(trajectory_directory / file_name)
        img = img.resize((640, 480))
        draw = ImageDraw.Draw(img)
        text = f"Episode {i}"
        text_length = draw.textlength(text, font=font)

        draw.text((img.width//2-text_length//2, 12),
                  text=text, font=font, fill=(0, 0, 0))
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(cv_img)
        if i == n_frames:
            break

    video.release()


def create_evolution_plot(trajectory_directory, interval=15, width=6, height=10):
    files = os.listdir(trajectory_directory)
    img1 = Image.open(trajectory_directory / files[0])
    small_size = img1.size
    big_image = Image.new("RGB", (width * img1.width, height * img1.height))
    font = ImageFont.load_default(48)
    for y in range(height):
        for x in range(width):
            i = x + y * width
            small_image = Image.open(
                trajectory_directory / files[i * interval])
            draw = ImageDraw.Draw(small_image)
            text = f"Episode {i*interval+1}"
            text_length = draw.textlength(text, font)
            draw.text(xy=(small_image.width//2 - text_length//2, 20),
                      text=text, fill=(0, 0, 0), font=font)
            draw.rectangle((473, 18, 473 + 150, 18 + 43), fill=(255, 255, 255))
            big_image.paste(
                small_image,
                (x * small_size[0], y * small_size[1])
            )
    big_image.show()
    big_image.save("sac_evolution.png")


# create_evolution_plot(Path.cwd() / "sac" / "trajectories")

create_evolution_movie(Path.cwd() / "sac" / "trajectories")
# create_landing_plots()
