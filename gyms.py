from spacecraft import Environment
from copy import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from common import Settings
import matplotlib.pyplot as plt

# It is recommended to normalize the environment
# https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment


class Normalization:
    MEAN = np.array([
        0,  # delta x
        0,  # delta y
        0,  # velocity x
        -5,  # velocity y
        1,  # cos (angle)
        0,  # sin(angle)
        0,  # angular velocity
        5,  # thrust level
        0,  # gimbal level
    ])

    SD = np.array([
        100,  # delta x
        100,  # delta y
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

    def __init__(self, discrete_actions=True):
        super().__init__()
        self.env = Environment(time_step_size=Settings.TIME_STEP_SIZE)

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
            self.env.MIN_THRUST_LEVEL,  # thrust level
            self.env.MIN_GIMBAL_LEVEL  # gimbal level
        ])

        obs_high = np.array([
            max_dist,  # delta x
            max_dist,  # delta y
            max_velocity,  # x velocity
            max_velocity,  # y velocity
            1,  # cos angle
            1,  # sin angle
            max_angular_velocity,  # angular velocity
            self.env.MAX_THRUST_LEVEL,  # thrust level
            self.env.MAX_GIMBAL_LEVEL  # gimbal level
        ])

        obs_low = (obs_low - Normalization.MEAN) / Normalization.SD
        obs_high = (obs_high - Normalization.MEAN) / Normalization.SD

        self.observation_space = spaces.Box(
            obs_low, obs_high, dtype=np.float64)

    def render(self):
        pass

    def close(self):
        pass


class LandingSpacecraftGym(SpacecraftGym):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    def __init__(self, discrete_actions=True):
        super().__init__(discrete_actions=discrete_actions)
        # Distance away from the landing are that the spacecraft spawn:
        self.x_start = -100
        self.x_end = 100
        self.y_start = 100
        self.y_end = 200
        self.landing_area = (self.env.map.width//2, 10)
        self.reset()

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

        obs = self._get_obs()

        # Normalize obs
        obs = (obs - Normalization.MEAN) / Normalization.SD

        y_distance_to_landing_area = (
            self.env.position[1] - self.landing_area[1])
        early_termination = False
        has_landed = False
        if self.env.state == self.env.STATE_ENDED:
            distance_penalty = self.env.get_distance_to(*self.landing_area) * 2
            velocity_penalty = self.env.get_velocity() * 10
            angle_penalty = np.fabs(self.env.angle) * 15
            angular_velocity_penalty = np.fabs(self.env.angular_velocity) * 15
            reward = 1000 - (distance_penalty + angle_penalty +
                             velocity_penalty + angular_velocity_penalty)
            has_landed = True
        elif np.fabs(self.env.angular_velocity) > 0.5 or self.env.get_velocity() > 15 or np.fabs(self.env.angle) > 0.6:
            early_termination = True
            reward = -1000
        # elif (self.env.position[1] - self.landing_area[1]) < 64 and (np.fabs(self.env.angular_velocity) > 0.1 or np.fabs(self.env.angle) > 0.05 or self.env.get_velocity() > 3):
        #     early_termination = True
        #     reward = -1000
        else:
            # vector pointing towards jump above the landing site
            landing_area_vec = (np.array(
                [self.landing_area[0], self.landing_area[1] + 32]) - np.array(self.env.position))

            # normalize landing site vector
            landing_area_vec = landing_area_vec / \
                (np.linalg.norm(landing_area_vec) + + 0.0001)

            # normalized velocity vector
            velocity_vector = self.env.velocity / \
                (np.linalg.norm(self.env.velocity) + 0.0001)

            # guide vehicle towards landing area (Gaze heuristic)
            reward = -1
            reward -= np.linalg.norm(landing_area_vec-velocity_vector) * \
                self.env.get_distance_to(*self.landing_area) * 1e-1

            # reward -= self.env.get_distance_to(*self.landing_area) * 1e-2

            # Maybe add a positive but descending reward for staying in the air - that turns negative after a while
            # reward -= (self.env.get_velocity()**2) * 1e-2
            # reward -= np.fabs(self.env.angle)

        info = {
            "early_termination": early_termination,
            "landed": has_landed
        }

        terminated = has_landed or early_termination

        truncated = False
        if self.env.steps >= 2000:
            print("Warning: n steps >= 2000")
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset()

        # Position spacecraft at some random location above the landing area:
        self.env.position = (self.landing_area[0] + np.random.randint(self.x_start, self.x_end),
                             self.landing_area[1] + np.random.randint(self.y_start, self.y_end))

        # With a bit of velocity, angle and angular velocity
        self.env.velocity = (np.random.uniform(-5, 5),
                             np.random.uniform(-5, 5))
        self.env.angle = np.random.uniform(-np.pi/16, np.pi/16)
        self.env.angular_velocity = np.random.uniform(-0.1, 0.1)

        # And with a random rocket engine configuration
        self.env.thrust_level = np.random.randint(
            self.env.MIN_THRUST_LEVEL, self.env.MAX_THRUST_LEVEL)
        self.env.gimbal_level = np.random.randint(
            self.env.MIN_GIMBAL_LEVEL, self.env.MAX_GIMBAL_LEVEL)

        info = {}
        observation = self._get_obs()
        return observation, info


class LandingEvaluator:

    def __init__(self, n_x=4, n_y=3):
        self.gym = LandingSpacecraftGym()
        self.x_positions = np.linspace(
            self.gym.x_start, self.gym.x_end, num=n_x)
        self.y_positions = np.linspace(
            self.gym.y_start, self.gym.y_end, num=n_y)
        self.idx = (0, 0)
        self.results = None

    def _set_env(self, x, y):
        self.gym.env.reset()
        self.gym.env.position = (self.gym.landing_area[0] + x,
                                 self.gym.landing_area[1] + y)
        return self.gym._get_obs()

    def get_avg_reward(self):
        reward_sum = 0
        for k, v in self.results.items():
            reward_sum += v["total_reward"]

        return reward_sum / len(self.results)

    def save_flight_trajectory_plot(self, path):
        plt.clf()
        plt.xlim((100, 500))
        plt.ylim((0, 300))
        for k, v in self.results.items():
            x_s = []
            y_s = []
            for x, y in v["flight_path"]:
                x_s.append(x)
                y_s.append(y)
            plt.plot(x_s, y_s)
            if v["early_termination"]:
                plt.scatter(x_s[-1], y_s[-1], marker="x")
            elif v["landed"]:
                plt.scatter(x_s[-1], y_s[-1], marker="*")

        plt.scatter(
            self.gym.landing_area[0], self.gym.landing_area[1] + 32, label="Landing area")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig(path)

    def evaluate(self, state_action_fn):
        results = {}
        for x in self.x_positions:
            for y in self.y_positions:
                obs = self._set_env(x, y)
                done = False
                key = (x, y)
                results[key] = {}
                results[key]["start"] = (x, y)
                results[key]["flight_path"] = [self.gym.env.position]
                results[key]["total_reward"] = 0
                while not done:
                    action = state_action_fn(obs)
                    obs, reward, terminated, truncated, info = self.gym.step(
                        action)
                    results[key]["flight_path"].append(self.gym.env.position)
                    results[key]["total_reward"] += reward
                    results[key]["early_termination"] = info["early_termination"]
                    results[key]["landed"] = info["landed"]
                    done = terminated or truncated
        self.results = results
        return results


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
