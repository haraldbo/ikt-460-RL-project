import torch
import torch.nn as nn
import numpy as np
from gyms import HoveringSpacecraftGym
from spacecraft import Environment
from common import Settings
from collections import deque
import random


def selct_action(model, state, epsilon):
    if random.random() > epsilon:
        model(state)
    else:
        pass


def train():
    init_env = Environment(time_step_size=Settings.TIME_STEP_SIZE)

    env = HoveringSpacecraftGym(init_env, point=(
        init_env.WORLD_SIZE//2, init_env.WORLD_SIZE//2))

    discount_rate = 0.99
    memory = deque(maxlen=10_000)
    epsilon = 0.1
    batch_size = 32

    policy_net = nn.Sequential(
        nn.Linear(9, 32),  # 9 observation variables
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 9)  # 9 actions
    )

    target_net = nn.Sequential(
        nn.Linear(9, 32),  # 9 observation variables
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 9)  # 9 actions
    )

    obs, _ = env.reset()
    done = False
    while not done:
        # Sample some state, action, reward, new_state from the env

        pass


if __name__ == "__main__":
    train()
