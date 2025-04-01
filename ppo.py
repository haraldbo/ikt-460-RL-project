from gyms import LandingSpacecraftGym, SpacecraftGym, HoveringSpacecraftGym, Normalization
from spacecraft import Environment
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO, DDPG
import numpy as np
from common import Settings, Agent


class PPOLandingAgent(Agent):

    def __init__(self, landing_area):
        self.landing_area = landing_area
        self.action_space = LandingSpacecraftGym(
            env=Environment()).discrete_action_space
        self.model = PPO.load(Settings.PPO_LANDER_BEST /
                              "best_model", device="cpu")

    def get_action(self, env: Environment):
        state = np.array([
            (env.position[0] - self.landing_area[0]),  # delta x
            (env.position[1] - self.landing_area[1]),  # delta y
            env.velocity[0],  # x velocity
            env.velocity[1],  # y velocity
            np.cos(env.angle),  # cos angle
            np.sin(env.angle),  # sin angle
            env.angular_velocity,  # angular velocity
            env.thrust_level,  # thrust level
            env.gimbal_level  # gimbal level
        ])

        state = (state - Normalization.MEAN)/Normalization.SD
        action_idx, _ = self.model.predict(state, deterministic=True)
        return self.action_space[action_idx]

    def handle_event(self, event):
        pass

    def reset(self):
        pass

    def render(self, window):
        pass


class PPOHoveringAgent(Agent):

    def __init__(self, hovering_point):
        self.hovering_point = hovering_point
        self.action_space = SpacecraftGym(
            env=Environment()).discrete_action_space
        self.model = PPO.load(Settings.PPO_HOVERING_BEST /
                              "best_model", device="cpu")

    def get_action(self, env: Environment):
        state = np.array([
            (env.position[0] - self.hovering_point[0]),  # delta x
            (env.position[1] - self.hovering_point[1]),  # delta y
            env.velocity[0],  # x velocity
            env.velocity[1],  # y velocity
            np.cos(env.angle),  # cos angle
            np.sin(env.angle),  # sin angle
            env.angular_velocity,  # angular velocity
            env.thrust_level,  # thrust level
            env.gimbal_level  # gimbal level
        ])

        state = (state - Normalization.MEAN)/Normalization.SD
        action_idx, _ = self.model.predict(state, deterministic=True)
        return self.action_space[action_idx]

    def handle_event(self, event):
        pass

    def reset(self):
        pass

    def render(self, window):
        pass


def train_landing_agent(init_env: Environment):
    env = LandingSpacecraftGym(env=init_env)

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
        gamma=1
    )
    model.learn(total_timesteps=10_000_000, callback=[
                checkpoint_callback, eval_callback])


def train_hovering_agent(init_env: Environment):
    env = HoveringSpacecraftGym(env=init_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=Settings.PPO_HOVERING_CHECKPOINT,

        name_prefix="ppo_hovering",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=Settings.PPO_HOVERING_BEST,
        eval_freq=5000,
        n_eval_episodes=100,
        deterministic=True
    )

    model = PPO(
        "MlpPolicy", env,
        verbose=0,
        device="cpu",
        tensorboard_log="./tensorboard_logs",
        gamma=1
    )
    model.learn(total_timesteps=10_000_000, callback=[
                checkpoint_callback, eval_callback])


if __name__ == "__main__":
    init_env = Environment(time_step_size=Settings.TIME_STEP_SIZE)
    # train_landing_agent(init_env)
    train_hovering_agent(init_env)
