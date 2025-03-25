from gyms import LandingSpacecraftGymEnv, HoveringSpacecraftGymEnv
from spacecraft import Environment
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO, DDPG
from spacecraft_visualization import start_visualization
import numpy as np
from common import *


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
            change_amount = 20
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
        pass


def train_hovering_agent(env: Environment):

    # env = make_vec_env(lambda: HoveringSpacecraftGymEnv(point=(
    #    WORLD_SIZE//2, WORLD_SIZE//2), env=env), n_envs=4, vec_env_cls=SubprocVecEnv)
    env = HoveringSpacecraftGymEnv(
        env=env, point=(env.WORLD_SIZE//2, env.WORLD_SIZE//2))
    # check_env(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./saves/",
        name_prefix="ppo_spacecraft",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        device="cpu",
        tensorboard_log="./tensorboard_logs",
        gamma=0.9
    )
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)
    # model.save("ppo_spacecraft")


def test_hovering_agent(env: Environment):
    model = PPO.load("./saves/ppo_spacecraft_620000_steps", device="cpu")
    point = (env.WORLD_SIZE//2, env.WORLD_SIZE//2)
    agent = PPOHoveringAgent(model, point)
    env.position = point
    env.state = env.STATE_IN_FLIGHT
    start_visualization(env, fps=30, agent=agent,
                        save_animation_frames=False)


def train_landing_agent(init_env: Environment):
    env = LandingSpacecraftGymEnv(env=init_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=Settings.PPO_LANDER_CHECKPOINT,
        name_prefix="ppo_landing",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=Settings.PPO_LANDER_BEST,
        eval_freq=5000,
        n_eval_episodes=100,
        deterministic=True
    )

    model = PPO(
        "MlpPolicy", env,
        verbose=0,
        device="cpu",
        tensorboard_log="./tensorboard_logs",
        gamma=0.99
    )
    model.learn(total_timesteps=10_000_000, callback=[
                checkpoint_callback, eval_callback])


def test_landing_agent(env: Environment):
    model = PPO.load(Settings.PPO_LANDER_BEST / "best_model", device="cpu")
    point = (env.landing_area[0] - 0, env.landing_area[1] + 100)
    agent = PPOLandingAgent(model)
    env.position = point
    env.thrust_level = 5
    env.angular_velocity = 0.2
    env.state = env.STATE_IN_FLIGHT
    start_visualization(env, fps=30, agent=agent,
                        save_animation_frames=False)


if __name__ == "__main__":
    init_env = Environment(time_step_size=1/5)
    train_landing_agent(init_env)
    # test_landing_agent(init_env)
