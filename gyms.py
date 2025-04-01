from spacecraft import Environment
from copy import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# It is recommended to normalize the environment
# https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment

class Normalization:

    MEAN = np.array([
        0,  # delta x
        0,  # delta y
        0,  # velocity x
        0,  # velocity y
        0,  # cos (angle)
        0,  # sin(angle)
        0,  # angular velocity
        0,  # thrust level
        0,  # gimbal level
    ])

    SD = np.array([
        50,  # delta x
        50,  # delta y
        10,  # velocity x
        10,  # velocity y
        1,  # cos(angle)
        1,  # sin(angle)
        1,  # angular velocity
        5,  # thrust level
        5,  # gimbal level
    ])


class SpacecraftGym(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, env: Environment, discrete_actions=True):
        super().__init__()
        self.env = env

        self.discrete_actions = discrete_actions
        self.discrete_action_space = []
        for gimbal_action in [-1, 0, 1]:
            for thrust_action in [-1, 0, 1]:
                self.discrete_action_space.append(
                    (gimbal_action, thrust_action))

        if discrete_actions:
            """
            The actions:
                - thruster: decrease, do nothing, increase 
                - gimbal: decrease, do nothing, increase

            9 in total: [-1, -1], [0, -1], ... , [1, 1] 
            """
            self.action_space = spaces.Discrete(
                len(self.discrete_action_space))

        else:
            action_low = np.array([
                -1,  # gimbal
                -1  # thrust
            ])

            action_high = np.array([
                1,  # gimbal
                1  # thrust
            ])
            self.action_space = spaces.Box(
                action_low, action_high, dtype=np.float64)

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
        max_dist = 400

        obs_low = np.array([
            -max_dist,  # delta x
            -max_dist,  # delta y
            -max_velocity,  # x velocity
            -max_velocity,  # y velocity
            -1,  # cos angle
            -1,  # sin angle
            -max_angular_velocity,  # angular velocity
            env.MIN_THRUST_LEVEL,  # thrust level
            env.MIN_GIMBAL_LEVEL  # gimbal level
        ])

        obs_high = np.array([
            max_dist,  # delta x
            max_dist,  # delta y
            max_velocity,  # x velocity
            max_velocity,  # y velocity
            1,  # cos angle
            1,  # sin angle
            max_angular_velocity,  # angular velocity
            env.MAX_THRUST_LEVEL,  # thrust level
            env.MAX_GIMBAL_LEVEL  # gimbal level
        ])

        obs_low = (obs_low - Normalization.MEAN) / Normalization.SD
        obs_high = (obs_high - Normalization.MEAN) / Normalization.SD

        self.observation_space = spaces.Box(
            obs_low, obs_high, dtype=np.float64)
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

    def __init__(self, env, discrete_actions=True):
        self.landing_area = (env.map.width//2, 10)
        super().__init__(env, discrete_actions=discrete_actions)

    def _get_obs(self):
        state = [
            self.env.position[0] - self.landing_area[0],  # delta x
            self.env.position[1] - self.landing_area[1],  # delta y
            self.env.velocity[0],  # x velocity
            self.env.velocity[1],  # y velocity
            np.cos(self.env.angle),  # cos angle
            np.sin(self.env.angle),  # sin angle
            self.env.angular_velocity,  # angular velocity
            self.env.thrust_level,  # thrust level
            self.env.gimbal_level  # gimbal level
        ]
        return np.array(state, dtype=np.float64)

    def step(self, action_input):
        if self.discrete_actions:
            action = self.discrete_action_space[action_input]
        else:
            action = action_input

        self.env.step(action)

        new_obs = self._get_obs()

        new_obs = (new_obs - Normalization.MEAN) / Normalization.SD
        old_obs = (old_obs - Normalization.MEAN) / Normalization.SD

        flight_ended = self.env.state == self.env.STATE_ENDED

        if flight_ended:
            distance_penalty = self.env.get_distance_to(*self.landing_area)
            velocity_penalty = self.env.get_velocity() * 5
            angle_penalty = np.fabs(self.env.angle) * 15
            angular_velocity_penalty = np.fabs(self.env.angular_velocity) * 15
            reward = 100 - (distance_penalty + angle_penalty +
                            velocity_penalty + angular_velocity_penalty)
        else:
            reward = -0.01
            # Maybe add a positive but descending reward for staying in the air - that turns negative after a while
            reward -= self.env.get_distance_to(*self.landing_area) * 1e-2
            reward -= (self.env.get_velocity()**2) * 1e-2
            reward -= np.fabs(self.env.angle)

        info = {}

        terminated = flight_ended
        truncated = self.env.steps >= 200
        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()

        # Position spacecraft at some random location above the landing area:
        self.env.position = (self.landing_area[0] + np.random.randint(-20, 20),
                             self.landing_area[1] + np.random.randint(40, 80))

        # With a bit of velocity, angle and angular velocity
        self.env.velocity = (np.random.uniform(-5, 5),
                             np.random.uniform(-5, 5))
        self.env.angle = np.random.uniform(-np.pi/8, np.pi/8)
        self.env.angular_velocity = np.random.uniform(-0.1, 0.1)

        # And with a random rocket engine configuration
        self.env.thrust_level = np.random.randint(
            self.env.MIN_THRUST_LEVEL, self.env.MAX_THRUST_LEVEL)
        self.env.gimbal_level = np.random.randint(
            self.env.MIN_GIMBAL_LEVEL, self.env.MAX_GIMBAL_LEVEL)

        info = {}
        self.env.state = self.env.STATE_FLIGHT
        observation = self._get_obs()
        return observation, info


class HoveringSpacecraftGym(SpacecraftGym):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, env, discrete_actions=True):
        self.hovering_point = (env.map.width//2, env.map.height//2)
        super().__init__(env, discrete_actions=discrete_actions)

    def _get_obs(self):
        state = [
            self.env.position[0] - self.hovering_point[0],  # delta x
            self.env.position[1] - self.hovering_point[1],  # delta y
            self.env.velocity[0],  # x velocity
            self.env.velocity[1],  # y velocity
            np.cos(self.env.angle),  # cos angle
            np.sin(self.env.angle),  # sin angle
            self.env.angular_velocity,  # angular velocity
            self.env.thrust_level,  # thrust level
            self.env.gimbal_level  # gimbal level
        ]
        return np.array(state, dtype=np.float64)

    def step(self, action_input):
        if self.discrete_actions:
            action = self.discrete_action_space[action_input]
        else:
            action = action_input

        self.env.step(action)

        new_obs = self._get_obs()
        new_obs = (new_obs - Normalization.MEAN) / Normalization.SD

        flight_ended = self.env.state == self.env.STATE_ENDED

        if flight_ended:
            reward = -10000
        else:
            distance_penalty = self.env.get_distance_to(
                *self.hovering_point) * 1e-1
            velocity_penalty = self.env.get_velocity() * 1e-1
            angle_penalty = np.fabs(self.env.angle) * 10
            angular_velocity_penalty = np.fabs(self.env.angular_velocity) * 10
            reward = 100 - (distance_penalty + velocity_penalty +
                            angle_penalty + angular_velocity_penalty)

        info = {}

        terminated = flight_ended
        truncated = self.env.steps >= 100
        return new_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()

        # Position spacecraft at some random location near the point:
        self.env.position = (self.hovering_point[0] + np.random.randint(-50, 50),
                             self.hovering_point[1] + np.random.randint(-50, 50))

        # With a bit of velocity, angle and angular velocity:
        self.env.velocity = (np.random.uniform(-10, 10),
                             np.random.uniform(-10, 10))
        self.env.angle = np.random.uniform(-np.pi/8, np.pi/8)
        self.env.angular_velocity = np.random.uniform(-0.2, 0.2)

        # And with a random engine gimbal/thrust configuration:
        self.env.thrust_level = np.random.randint(
            self.env.MIN_THRUST_LEVEL, self.env.MAX_THRUST_LEVEL)
        self.env.gimbal_level = np.random.randint(
            self.env.MIN_GIMBAL_LEVEL, self.env.MAX_GIMBAL_LEVEL)

        info = {}
        self.env.state = self.env.STATE_FLIGHT
        observation = self._get_obs()
        return observation, info
