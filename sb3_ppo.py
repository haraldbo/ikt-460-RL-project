from spacecraft import Environment, WORLD_SIZE, MAX_GIMBAL_LEVEL, MAX_THRUST_LEVEL, MIN_GIMBAL_LEVEL, MIN_THRUST_LEVEL, STATE_LAUNCH
from copy import copy
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from spacecraft_visualization import start_visualization, Agent
import pygame

# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
# "To use the RL baselines with custom environments, they just need to follow the gymnasium interface."

# Gymnasium interface https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
FPS = 1


class HoveringSpacecraftGymEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, point: tuple):
        super().__init__()
        self.point = point
        self.env = Environment()

        """
        The actions:
            - thruster: decrease, do nothing, increase 
            - gimbal: decrease, do nothing, increase

        9 in total: [-1, -1], [0, -1], ... , [1, 1] 
        """
        self.action_space = spaces.Discrete(len(self.env.action_space))

        """
        The observation:
            - delta x: distance from x position to target x along the x axis
            - delta y: distance from y position to target y along the y axis
            - x velocity
            - y velocity
            - sin(angle)
            - cos(angle)
            - angular velocity
            - thrust level
            - gimbal level
        """
        max_velocity = 100

        low = np.array([
            -WORLD_SIZE,  # delta x
            -WORLD_SIZE,  # delta y
            -max_velocity,  # x velocity
            -max_velocity,  # y velocity
            -1,  # cos angle
            -1,  # sin angle
            -2 * np.pi,  # angular velocity
            MIN_THRUST_LEVEL,  # thrust level
            MIN_GIMBAL_LEVEL  # gimbal level
        ])

        high = np.array([
            WORLD_SIZE,  # delta x
            WORLD_SIZE,  # delta y
            max_velocity,  # x velocity
            max_velocity,  # y velocity
            1,  # cos angle
            1,  # sin angle
            2 * np.pi,  # angular velocity
            MAX_THRUST_LEVEL,  # thrust level
            MAX_GIMBAL_LEVEL  # gimbal level
        ])

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.reset()

    def _get_obs(self):
        state = [
            self.env.position[0] - self.point[0],  # delta x
            self.env.position[1] - self.point[1],  # delta y
            self.env.velocity[0],  # x velocity
            self.env.velocity[1],  # y velocity
            np.cos(self.env.angle),  # cos angle
            np.sin(self.env.angle),  # sin angle
            self.env.angular_velocity,  # angular velocity
            self.env.thrust_level,  # thrust level
            self.env.gimbal_level  # gimbal level
        ]
        return np.array(state, dtype=np.float64)

    def _calculate_reward(self, current_env: Environment, previous_env: Environment):
        # Closer is better
        distance = math.sqrt((current_env.position[0] - self.point[0]) ** 2 + (
            current_env.position[1] - self.point[1]) ** 2)
        rotation = 20 * current_env.angular_velocity ** 2
        return 1/(distance + rotation + 1)

    def step(self, action_idx):
        action = self.env.action_space[action_idx]
        previous_env = copy(self.env)
        terminated = self.env.step(action)
        observation = self._get_obs()
        truncated = self.env.steps > 1000
        if terminated:
            reward = -10
        else:
            reward = self._calculate_reward(self.env, previous_env)
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.env.position = (
            self.point[0] + np.random.randint(-30, 30), self.point[1] + np.random.randint(-30, 30))

        self.env.velocity = (np.random.uniform(-20, 20),
                             np.random.uniform(-20, 20))
        self.env.angular_velocity = np.random.uniform(-0.4, 0.4)
        self.env.thrust_level = np.random.randint(
            MIN_THRUST_LEVEL, MAX_THRUST_LEVEL)
        self.env.gimbal_level = np.random.randint(
            MIN_GIMBAL_LEVEL, MAX_GIMBAL_LEVEL)
        info = {}
        observation = self._get_obs()
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


class PPOHoveringAgent(Agent):

    def __init__(self, model: PPO, point: tuple):
        super().__init__()
        self.model = model
        self.initial_point = point
        self.point = point

    def get_action(self, env: Environment):
        state = np.array([
            env.position[0] - self.point[0],  # delta x
            env.position[1] - self.point[1],  # delta y
            env.velocity[0],  # x velocity
            env.velocity[1],  # y velocity
            np.cos(env.angle),  # cos angle
            np.sin(env.angle),  # sin angle
            env.angular_velocity,  # angular velocity
            env.thrust_level,  # thrust level
            env.gimbal_level  # gimbal level
        ])
        action_idx, _ = self.model.predict(state, deterministic=True)
        return env.action_space[action_idx.item()]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            y_change = 0
            x_change = 0
            change_amount = 3
            if event.key == pygame.K_UP:
                y_change += change_amount
            if event.key == pygame.K_DOWN:
                y_change -= change_amount
            if event.key == pygame.K_LEFT:
                x_change -= change_amount
            if event.key == pygame.K_RIGHT:
                x_change += change_amount

            self.point = (self.point[0] + x_change, self.point[1] + y_change)

    def reset(self):
        self.point = self.initial_point

    def render(self, window):
        pygame.draw.circle(window, color=(0, 255, 0), center=(
            self.point[0], WORLD_SIZE-self.point[1]), radius=2)


def train_agent():
    # env = HoveringSpacecraftGymEnv(point=(400, 400))
    env = make_vec_env(lambda: HoveringSpacecraftGymEnv(
        point=(400, 400)), n_envs=8, vec_env_cls=SubprocVecEnv)
    # check_env(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./saves/",
        name_prefix="ppo_spacecraft",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model = PPO("MlpPolicy", env, verbose=1, batch_size=128,
                n_steps=12800, device="cpu", tensorboard_log="./tensorboard_logs")
    model.learn(total_timesteps=1000_000_000, callback=checkpoint_callback)
    # model.save("ppo_spacecraft")


def test_agent():
    model = PPO.load("./saves/ppo_spacecraft_7120000_steps", device="cpu")
    agent = PPOHoveringAgent(model, (470, 160))
    env = Environment()
    start_visualization(env, fps=60, agent=agent,
                        save_animation_frames=True)


if __name__ == "__main__":
    # train_agent()
    test_agent()
