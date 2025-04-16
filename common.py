from collections import deque
import random
import torch
import numpy as np


class Settings:
    TIME_STEP_SIZE = 1/10
    SIMULATION_FPS = 60
    SIMULATION_FRAME_SIZE = (800, 800)
    RENDERING_VIEWPORT_SIZE = (256, 256)
    RENDERING_SPACECRAFT_DEBUGGING = False
    RENDER_SPACECRAFT_INFORMATION = True


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add_transition(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_lst = np.array(s_lst)
        s_prime_lst = np.array(s_prime_lst)
        a_lst = np.array(a_lst)

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)
