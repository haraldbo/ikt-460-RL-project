from spacecraft import Environment
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


FPS = 1


class LandingSpacecraftGymEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env

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

        # Probably won't reach this values
        max_velocity = 200
        max_angular_velocity = 2 * np.pi

        low = np.array([
            -env.WORLD_SIZE,  # delta x
            -env.WORLD_SIZE,  # delta y
            -max_velocity,  # x velocity
            -max_velocity,  # y velocity
            -1,  # cos angle
            -1,  # sin angle
            -max_angular_velocity,  # angular velocity
            env.MIN_THRUST_LEVEL,  # thrust level
            env.MIN_GIMBAL_LEVEL  # gimbal level
        ])

        high = np.array([
            env.WORLD_SIZE,  # delta x
            env.WORLD_SIZE,  # delta y
            max_velocity,  # x velocity
            max_velocity,  # y velocity
            1,  # cos angle
            1,  # sin angle
            max_angular_velocity,  # angular velocity
            env.MAX_THRUST_LEVEL,  # thrust level
            env.MAX_GIMBAL_LEVEL  # gimbal level
        ])

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.reset()

    def _get_obs(self):
        state = [
            self.env.position[0] - self.env.landing_area[0],  # delta x
            self.env.position[1] - self.env.landing_area[1],  # delta y
            self.env.velocity[0],  # x velocity
            self.env.velocity[1],  # y velocity
            np.cos(self.env.angle),  # cos angle
            np.sin(self.env.angle),  # sin angle
            self.env.angular_velocity,  # angular velocity
            self.env.thrust_level,  # thrust level
            self.env.gimbal_level  # gimbal level
        ]
        return np.array(state, dtype=np.float64)

    def step(self, action_idx):
        action = self.env.action_space[action_idx]
        self.env.step(action)
        flight_ended = self.env.state == self.env.STATE_ENDED
        observation = self._get_obs()

        if flight_ended:
            distance_penalty = self.env.get_distance_to_landing_site()
            velocity_penalty = self.env.get_velocity()
            angle_penalty = np.fabs(self.env.angle) * 20
            angular_velocity_penalty = np.fabs(self.env.angular_velocity) * 20
            reward = 100 - (distance_penalty + angle_penalty +
                            velocity_penalty + angular_velocity_penalty)
        else:
            reward = -0.01
        info = {}

        terminated = flight_ended
        truncated = self.env.steps >= 100

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.env.position = (self.env.landing_area[0] + np.random.randint(-40, 40),
                             self.env.landing_area[1] + np.random.randint(40, 80))

        self.env.velocity = (np.random.uniform(-5, 5),
                             np.random.uniform(-5, 5))
        self.env.angle = np.random.uniform(-np.pi/4, np.pi/4)
        self.env.angular_velocity = np.random.uniform(-0.3, 0.3)
        self.env.thrust_level = np.random.randint(
            self.env.MIN_THRUST_LEVEL, self.env.MAX_THRUST_LEVEL)
        self.env.gimbal_level = np.random.randint(
            self.env.MIN_GIMBAL_LEVEL, self.env.MAX_GIMBAL_LEVEL)

        info = {}
        self.env.state = self.env.STATE_IN_FLIGHT
        observation = self._get_obs()
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


class PPOLandingAgent(Agent):

    def __init__(self, model: PPO):
        super().__init__()
        self.model = model

    def get_action(self, env: Environment):
        state = np.array([
            env.position[0] - env.landing_area[0],  # delta x
            env.position[1] - env.landing_area[1],  # delta y
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
        pass

    def reset(self):
        pass

    def render(self, window):
        pass


def train_agent(init_env: Environment):

    # env = make_vec_env(lambda: HoveringSpacecraftGymEnv(point=(
    #    WORLD_SIZE//2, WORLD_SIZE//2), env=env), n_envs=4, vec_env_cls=SubprocVecEnv)
    env = LandingSpacecraftGymEnv(env=init_env)
    # check_env(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./saves/",
        name_prefix="ppo_landing",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        device="cpu",
        tensorboard_log="./tensorboard_logs",
        gamma=0.99
    )
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)
    # model.save("ppo_spacecraft")


def test_agent(env: Environment):
    model = PPO.load("./saves/ppo_landing_610000_steps", device="cpu")
    point = (env.landing_area[0] - 10, env.landing_area[1] + 50)
    agent = PPOLandingAgent(model)
    env.position = point
    env.state = env.STATE_IN_FLIGHT
    start_visualization(env, fps=10, agent=agent,
                        save_animation_frames=False)


if __name__ == "__main__":
    init_env = Environment(time_step_size=1/5)
    train_agent(init_env)
    # test_agent(init_env)
