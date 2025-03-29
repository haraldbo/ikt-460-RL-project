from spacecraft import Environment
from copy import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# It is recommended to normalize the environment
# https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
class Normalization:
    # These mean and sd values were found by gathering running mean/variance during training of an agent with the PPO algorithm.
    # - They are likely dependent on the initial configuration of the environment (See the reset method) and algorithm.
    # - They seem to work quite well, but that is all. They should not be used for anything serious.
    class Landing:
        MEAN = np.array([
            0,  # delta x
            50,  # delta y
            0,  # x velocity
            -2,  # y velocity
            1,  # cos (angle)
            0,  # sin(angle)
            0,  # angular velocity
            3,  # thrust level
            0,  # gimbal level
        ])

        SD = np.array([
            45,  # delta x
            19,  # delta y
            8,  # delta x
            5,  # delta y
            0.2,  # cos(angle)
            0.3,  # sin(angle)
            0.14,  # angular velocity
            1.14,  # thrust level
            2,  # gimbal level
        ])

    class Hoovering:
        pass

    POSITION = 100
    VELOCITY = 50
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
        max_dist = 400

        low = np.array([
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

        high = np.array([
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

        low = (low - Normalization.Landing.MEAN)/Normalization.Landing.SD
        high = (high - Normalization.Landing.MEAN)/Normalization.Landing.SD

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

    def _calculate_reward(self, old_obs: np.ndarray, new_obs: np.ndarray, flight_ended: bool):
        old_distance = np.sqrt(old_obs[0] ** 2 + old_obs[1] ** 2)
        new_distance = np.sqrt(new_obs[0] ** 2 + new_obs[1] ** 2)

        old_velocity = np.sqrt(old_obs[2] ** 2 + old_obs[3] ** 2)
        new_velocity = np.sqrt(new_obs[2] ** 2 + new_obs[3] ** 2)

        old_angular_velocity = old_obs[6]
        new_angular_velocity = new_obs[6]

        new_angle = np.arctan2(new_obs[5], new_obs[4])

        delta_velocity = new_velocity - old_velocity
        delta_angular_velocity = new_angular_velocity - old_angular_velocity
        delta_distance = new_distance - old_distance

        reward = 0
        reward -= 100 * delta_distance
        reward -= 100 * delta_velocity
        reward -= 100 * delta_angular_velocity

        if flight_ended:
            reward += 100

            reward -= 100 * new_distance
            reward -= 100 * new_velocity
            reward -= 100 * new_angular_velocity
            reward -= 100 * new_angle

        return reward

    def step(self, action_idx):
        action = self.env.action_space[action_idx]
        old_obs = self._get_obs()
        self.env.step(action)

        new_obs = self._get_obs()

        new_obs = (new_obs - Normalization.Landing.MEAN) / \
            Normalization.Landing.SD
        old_obs = (old_obs - Normalization.Landing.MEAN) / \
            Normalization.Landing.SD

        flight_ended = self.env.state == self.env.STATE_ENDED
        # reward = self._calculate_reward(old_obs, new_obs, flight_ended)

        if flight_ended:
            distance_penalty = self.env.get_distance_to(*self.landing_area)
            velocity_penalty = self.env.get_velocity() * 5
            angle_penalty = np.fabs(self.env.angle) * 15
            angular_velocity_penalty = np.fabs(self.env.angular_velocity) * 15
            reward = 100 - (distance_penalty + angle_penalty +
                            velocity_penalty + angular_velocity_penalty)
        else:
            reward = 0
            # Maybe add a positive but descent reward for staying in the air - that turns negative after a while
            reward -= self.env.get_distance_to(*self.landing_area) * 1e-2
            reward -= self.env.get_velocity() * 1e-1
            reward -= np.fabs(self.env.angle) * 10

            # reward -= self.env.velocity[1] / \
            #    (np.fabs(self.env.position[1] - self.landing_area[1]) + 0.5)

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
