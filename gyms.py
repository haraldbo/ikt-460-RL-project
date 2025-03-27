from spacecraft import Environment
from copy import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# It is recommended to normalize the environment
# https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
class Scalers:
    POSITION = 100
    VELOCITY = 30
    THRUST = Environment.MAX_THRUST_LEVEL
    GIMBAL = Environment.MAX_GIMBAL_LEVEL


class SpacecraftGym(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
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

        # Probably won't exceed these values
        max_velocity = 60
        max_angular_velocity = 2 * np.pi
        max_dist = 200

        low = np.array([
            -max_dist/Scalers.POSITION,  # delta x
            -max_dist/Scalers.POSITION,  # delta y
            -max_velocity/Scalers.VELOCITY,  # x velocity
            -max_velocity/Scalers.VELOCITY,  # y velocity
            -1,  # cos angle
            -1,  # sin angle
            -max_angular_velocity,  # angular velocity
            env.MIN_THRUST_LEVEL/Scalers.THRUST,  # thrust level
            env.MIN_GIMBAL_LEVEL/Scalers.GIMBAL  # gimbal level
        ])

        high = np.array([
            max_dist/Scalers.POSITION,  # delta x
            max_dist/Scalers.POSITION,  # delta y
            max_velocity/Scalers.VELOCITY,  # x velocity
            max_velocity/Scalers.VELOCITY,  # y velocity
            1,  # cos angle
            1,  # sin angle
            max_angular_velocity,  # angular velocity
            env.MAX_THRUST_LEVEL/Scalers.THRUST,  # thrust level
            env.MAX_GIMBAL_LEVEL/Scalers.GIMBAL  # gimbal level
        ])

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.reset()

    def render(self):
        pass

    def close(self):
        pass


class LandingSpacecraftGym(SpacecraftGym):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, env):
        self.landing_area = (env.map.width//2, 10)
        super().__init__(env)

    def _get_obs(self):
        state = [
            (self.env.position[0] - self.landing_area[0]) /
            Scalers.POSITION,  # delta x
            (self.env.position[1] - self.landing_area[1]) /
            Scalers.POSITION,  # delta y
            (self.env.velocity[0])/Scalers.VELOCITY,  # x velocity
            (self.env.velocity[1])/Scalers.VELOCITY,  # y velocity
            np.cos(self.env.angle),  # cos angle
            np.sin(self.env.angle),  # sin angle
            self.env.angular_velocity,  # angular velocity
            self.env.thrust_level/Scalers.THRUST,  # thrust level
            self.env.gimbal_level/Scalers.GIMBAL  # gimbal level
        ]
        return np.array(state, dtype=np.float64)

    def step(self, action_idx):
        action = self.env.action_space[action_idx]
        self.env.step(action)
        flight_ended = self.env.state == self.env.STATE_ENDED
        observation = self._get_obs()

        # Landing reward

        if flight_ended:
            distance_penalty = self.env.get_distance_to(
                self.env.map.width//2, 10)
            velocity_penalty = self.env.get_velocity() * 5
            angle_penalty = np.fabs(self.env.angle) * 15
            angular_velocity_penalty = np.fabs(self.env.angular_velocity) * 15
            reward = 100 - (distance_penalty + angle_penalty +
                            velocity_penalty + angular_velocity_penalty)
        else:
            reward = -0.05

        info = {}

        terminated = flight_ended
        truncated = self.env.steps >= 200

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()

        # Position spacecraft at some random location above the landing area:
        self.env.position = (self.landing_area[0] + np.random.randint(-20, 20),
                             self.landing_area[1] + np.random.randint(40, 80))

        # With a bit of velocity, angle and angular velocity
        self.env.velocity = (np.random.uniform(-10, 10),
                             np.random.uniform(-10, 10))
        self.env.angle = np.random.uniform(-np.pi/4, np.pi/4)
        self.env.angular_velocity = np.random.uniform(-0.3, 0.3)

        # And with a random rocket engine configuration
        self.env.thrust_level = np.random.randint(
            self.env.MIN_THRUST_LEVEL, self.env.MAX_THRUST_LEVEL)
        self.env.gimbal_level = np.random.randint(
            self.env.MIN_GIMBAL_LEVEL, self.env.MAX_GIMBAL_LEVEL)

        info = {}
        self.env.state = self.env.STATE_FLIGHT
        observation = self._get_obs()
        return observation, info
