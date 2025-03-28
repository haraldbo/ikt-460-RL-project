from gyms import LandingSpacecraftGym, Normalization
from spacecraft import Environment
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import DQN
import numpy as np
from common import Settings, Agent


class DQNLandingAgent(Agent):

    def __init__(self, model: DQN):
        super().__init__()
        self.model = model

    def get_action(self, env: Environment):
        state = np.array([
            (env.position[0] - env.map.width//2),  # delta x
            (env.position[1] - 10),  # delta y
            env.velocity[0],  # x velocity
            env.velocity[1],  # y velocity
            np.cos(env.angle),  # cos angle
            np.sin(env.angle),  # sin angle
            env.angular_velocity,  # angular velocity
            env.thrust_level,  # thrust level
            env.gimbal_level  # gimbal level
        ])

        state = (state - Normalization.Landing.MEAN) / Normalization.Landing.SD
        action_idx, _ = self.model.predict(state, deterministic=True)
        return env.action_space[action_idx.item()]

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
        save_path=Settings.DQN_LANDER_CHECKPOINT,
        name_prefix="dqn_landing",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=Settings.DQN_LANDER_BEST,
        eval_freq=5000,
        n_eval_episodes=100,
        deterministic=True
    )

    model = DQN(
        "MlpPolicy", env,
        verbose=0,
        device="cpu",
        tensorboard_log="./tensorboard_logs",
        gamma=1,
        buffer_size=4096 * 8,
        batch_size=128,
        train_freq=4,
        target_update_interval=4096,
        tau=0.5,
        exploration_final_eps=0.01,
        exploration_fraction=0.5
    )
    model.learn(total_timesteps=2_000_000, callback=[
                checkpoint_callback, eval_callback])


if __name__ == "__main__":
    init_env = Environment(time_step_size=Settings.TIME_STEP_SIZE)
    train_landing_agent(init_env)
