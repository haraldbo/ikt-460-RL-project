from collections import deque
import random
import torch
import numpy as np


class Settings:
    TIME_STEP_SIZE = 1/5
    SIMULATION_FPS = 25
    SIMULATION_FRAME_SIZE = (600, 600)
    RENDERING_VIEWPORT_SIZE = (256, 256)
    RENDERING_SPACECRAFT_DEBUGGING = False
    RENDER_SPACECRAFT_INFORMATION = True


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add_transition(self, transition):
        self.buffer.append(transition)

    def create_batch(self, n):
        sample = random.sample(self.buffer, n)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for transition in sample:
            s, a, r, s_prime, done = transition
            states.append(s)
            actions.append(a)
            rewards.append([r])
            next_states.append(s_prime)
            dones.append([0.0 if done else 1.0])

        return torch.tensor(np.array(states), dtype=torch.float), torch.tensor(np.array(actions), dtype=torch.float), \
            torch.tensor(np.array(rewards), dtype=torch.float), torch.tensor(np.array(next_states), dtype=torch.float), \
            torch.tensor(dones, dtype=torch.float)

    def size(self):
        return len(self.buffer)
