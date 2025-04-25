from spacecraft import Environment
import numpy as np
from common import Settings
import matplotlib.pyplot as plt


# "It is recommended to normalize the environment" - https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
# it makes sense since it is going to be inputted to a neural network
class Normalization:
    MEAN = np.array([
        0,  # delta x
        0,  # delta y
        0,  # velocity x
        0,  # velocity y
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
        0.2,  # angular velocity
        5,  # thrust level
        3,  # gimbal level
    ])


def create_normalized_observation(env: Environment, target_point):
    state = np.array([
        env.position[0] - target_point[0],  # delta x
        env.position[1] - target_point[1],  # delta y
        env.velocity[0],  # x velocity
        env.velocity[1],  # y velocity
        np.cos(env.angle),  # cos angle
        np.sin(env.angle),  # sin angle
        env.angular_velocity,  # angular velocity
        env.thrust_level,  # thrust level
        env.gimbal_level  # gimbal level
    ], dtype=np.float64)

    state = (state - Normalization.MEAN) / Normalization.SD

    return state


def mirror_observation(obs: np.ndarray):
    new_obs = np.copy(obs)
    new_obs[0] *= -1
    new_obs[2] *= -1
    new_obs[5] *= -1
    new_obs[6] *= -1
    new_obs[8] *= -1
    return new_obs


class LandingSpacecraftGym:

    def __init__(self, discrete_actions=True, relaxed_constraints = False):
        self.env = Environment(time_step_size=Settings.TIME_STEP_SIZE)

        self.discrete_actions = discrete_actions
        self.discrete_action_space = []
        for gimbal_action in [-1, 0, 1]:
            for thrust_action in [-1, 0, 1]:
                self.discrete_action_space.append(
                    (gimbal_action, thrust_action))

        # Distance away from the landing area that the spacecraft spawn:
        self.x_start = -150
        self.x_end = 150
        self.y_start = 100
        self.y_end = 300
        self.landing_area = (self.env.map.width//2, 10)
        self.target_point = (
            self.landing_area[0], self.landing_area[1] + self.env.height//2)
        if relaxed_constraints:
            self.landing_velocity = 5
            self.landing_angular_velocity = 0.5
            self.landing_angle = 0.5
        else:
            self.landing_velocity = 1
            self.landing_angular_velocity = 0.1
            self.landing_angle = 0.1
        
        self.reset()

    def step(self, action_input):
        if self.discrete_actions:
            action = self.discrete_action_space[action_input]
        else:
            action = action_input

        self.env.step(action)

        # approximate. If ship lands with an angle, side may hold a bit of a positive value
        y_distance_to_landing = np.fabs(
            self.env.position[1] - self.target_point[1])

        max_accepted_velocity = self.landing_velocity + y_distance_to_landing/100 * 10
        max_accepted_angular_velocity = self.landing_angular_velocity + y_distance_to_landing/100 * 0.5
        max_accepted_angle = self.landing_angle + y_distance_to_landing/100 * 0.5

        terminated = False
        has_landed = False

        x_distance = np.fabs(self.env.position[0] - self.target_point[0])

        if self.env.state == self.env.STATE_ENDED:
            if y_distance_to_landing < 10:
                reward = 200 - x_distance
                has_landed = True
            else:
                reward = -100
                terminated = True
        elif np.fabs(self.env.angular_velocity) > max_accepted_angular_velocity or self.env.get_velocity() > max_accepted_velocity or np.fabs(self.env.angle) > max_accepted_angle:
            terminated = True
            reward = -100
        else:
            # use gaze heuristic to guide the vehicle towards the landing area:

            # vector pointing from bottom of spacecraft towards landing area
            landing_area_unit_vec = np.array(
                self.target_point) - np.array(self.env.position)

            # unit vector point torward landing area
            landing_area_unit_vec = landing_area_unit_vec / \
                np.linalg.norm(landing_area_unit_vec)

            # unit velocity vector
            velocity_unit_vector = np.array(self.env.velocity) / \
                np.linalg.norm(self.env.velocity)

            # length of difference vector should do
            # it is in range [0, 2]
            direction_error = np.linalg.norm(
                velocity_unit_vector - landing_area_unit_vec)

            # sqrt(2-2cos(pi / 24)) = 0.13, 7.5 degrees off
            # sqrt(2-2cos(pi / 12)) = 0.261, 15 degrees off
            reward = 0.261-direction_error

        info = {
            "terminated": terminated,
            "landed": has_landed
        }

        flight_ended = has_landed or terminated

        truncated = False
        if self.env.steps >= 1000:
            # print("Warning: n steps >= 1000")
            truncated = True

        obs = create_normalized_observation(self.env, self.target_point)
        return obs, reward, flight_ended, truncated, info

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
        observation = create_normalized_observation(
            self.env, self.target_point)
        return observation, info


class LandingEvaluator:

    def __init__(self, n_x=4, n_y=3, discrete_actions=True):
        self.gym = LandingSpacecraftGym(discrete_actions=discrete_actions)
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
        return create_normalized_observation(self.gym.env, self.gym.target_point)

    def get_avg_reward(self):
        reward_sum = 0
        for k, v in self.results.items():
            reward_sum += v["total_reward"]

        return reward_sum / len(self.results)

    def get_avg_episode_length(self):
        length_sum = 0
        for k, v in self.results.items():
            length_sum += len(v["flight_path"]) - 1
        return length_sum / len(self.results)

    def get_num_landings(self):
        n_landings = 0
        for k, v in self.results.items():
            if v["landed"]:
                n_landings += 1
        return n_landings

    def print_results(self):
        print(f"Average reward: {self.get_avg_reward()}")
        print(f"Average length: {self.get_avg_episode_length()}")
        print(
            f"Successful landings: {self.get_num_landings()} / {len(self.results)}")

    def save_flight_trajectory_plot(self, path):
        plt.clf()
        plt.xlim((0, 400))
        plt.ylim((0, 400))
        for k, v in self.results.items():
            x_s = []
            y_s = []
            for x, y in v["flight_path"]:
                x_s.append(x)
                y_s.append(y)

            color = "green" if v["landed"] else "red"
            plt.plot(x_s, y_s, color=color)
            if v["terminated"]:
                plt.scatter(x_s[-1], y_s[-1], color=color, marker="x")
            elif v["landed"]:
                plt.scatter(x_s[-1], y_s[-1], color=color, marker="*")

            plt.scatter(x_s[0], y_s[0], color="black", marker="o")

        plt.scatter([self.gym.landing_area[0]], [
                    self.gym.landing_area[1] + 32], s=[500], label="Landing area", zorder=-1)
        plt.xlabel("x")
        plt.ylabel("y")

        # get current axes
        ax = plt.gca()

        # hide x-axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.legend()
        plt.tight_layout()
        plt.savefig(path)

    def append_to_csv(self, ep, csv_path):
        with open(csv_path, "+a") as csv:
            csv.write(
                f"{ep},{self.get_avg_reward()},{self.get_avg_episode_length()},{self.get_num_landings()}\n"
            )

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
                    results[key]["terminated"] = info["terminated"]
                    results[key]["landed"] = info["landed"]
                    done = terminated or truncated
        self.results = results
        return results
