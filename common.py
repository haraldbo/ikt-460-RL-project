from abc import ABC, abstractmethod
from spacecraft import Environment
import pygame
from pathlib import Path


class Settings:
    TIME_STEP_SIZE = 1/2

    SAVES_DIR = Path(__file__).parent / "saves"

    PPO_SAVE_DIR = SAVES_DIR / "ppo"
    PPO_LANDER_BEST = PPO_SAVE_DIR / "lander" / "best"
    PPO_LANDER_CHECKPOINT = PPO_SAVE_DIR / "lander" / "checkpoint"
    PPO_HOVERING_BEST = PPO_SAVE_DIR / "hovering" / "best"
    PPO_HOVERING_CHECKPOINT = PPO_SAVE_DIR / "hovering" / "checkpoint"

    DQN_SAVE_DIR = SAVES_DIR / "dqn"
    DQN_LANDER_BEST = DQN_SAVE_DIR / "lander" / "best"
    DQN_LANDER_CHECKPOINT = DQN_SAVE_DIR / "lander" / "checkpoint"
    DQN_HOVERING_BEST = DQN_SAVE_DIR / "hovering" / "best"
    DQN_HOVERING_CHECKPOINT = DQN_SAVE_DIR / "hovering" / "checkpoint"

    DDPG_SAVE_DIR = SAVES_DIR / "ddpg"
    DDPG_LANDER_BEST = DDPG_SAVE_DIR / "lander" / "best"
    DDPG_LANDER_CHECKPOINT = DDPG_SAVE_DIR / "lander" / "checkpoint"
    DDPG_HOVERING_BEST = DDPG_SAVE_DIR / "hovering" / "best"
    DDPG_HOVERING_CHECKPOINT = DDPG_SAVE_DIR / "hovering" / "checkpoint"

    SIMULATION_FPS = 30
    SIMULATION_FRAME_SIZE = (400, 400)
    RENDERING_VIEWPORT_SIZE = (256, 256)
    RENDERING_SPACECRAFT_DEBUGGING = True
    RENDER_SPACECRAFT_INFORMATION = True


class Agent(ABC):

    @abstractmethod
    def get_action(self, environment: Environment):
        """
        Returns the next action to perform, given the current environment
        """
        pass
