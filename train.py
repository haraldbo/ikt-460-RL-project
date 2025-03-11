from spacecraft import Environment, WORLD_SIZE, MAX_GIMBAL_LEVEL, MAX_THRUST_LEVEL, MIN_GIMBAL_LEVEL, MIN_THRUST_LEVEL
from copy import copy
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np

# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
# "To use the RL baselines with custom environments, they just need to follow the gymnasium interface."

# Gymnasium interface https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
FPS = 1

class CustomEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self):
        super().__init__()

        """
        The actions:
            - thruster: decrease, do nothing, increase 
            - gimbal: decrease, do nothing, increase

        9 in total: [-1, -1], [0, -1], ... , [1, 1] 
        """
        self.action_space = spaces.Discrete(9)

        """
        The observation:
            - x coordinate
            - y coordinate
            - x velocity
            - y velocity
            - sin(angle)
            - cos(angle)
            - angular velocity
            - thrust level
            - gimbal level
        """
        max_velocity = 10

        low = np.array([
            0,  # x coordinate
            0,  # y coordinate
            -max_velocity,  # x velocity
            -max_velocity,  # y velocity
            -1,  # cos angle
            -1,  # sin angle
            -2 * np.pi,  # angular velocity
            MIN_THRUST_LEVEL,  # thrust level
            MIN_GIMBAL_LEVEL  # gimbal level
        ])

        high = np.array([
            WORLD_SIZE,  # x coordinate
            WORLD_SIZE,  # y coordinate
            max_velocity,  # x velocity
            max_velocity,  # y velocity
            1,  # cos angle
            1,  # sin angle
            2 * np.pi,  # angular velocity
            MAX_THRUST_LEVEL,  # thrust level
            MAX_GIMBAL_LEVEL  # gimbal level
        ])

        self.observation_space = spaces.Box(low, high)

    def step(self, action):
        pass
        # return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        pass
        # return observation, info

    def render(self):
        pass

    def close(self):
        pass


def reward(env_before, env_after):
    point = (400, 400)

    return -math.sqrt((env_after.position[0] - point[0]) ** 2 + (env_after.position[1] - point[1]) ** 2)


def main():
    env = Environment()
    print(env.position)
    env_before = copy(env)
    env.step((0, 0))
    print(reward(env_before, env))


if __name__ == "__main__":
    main()
