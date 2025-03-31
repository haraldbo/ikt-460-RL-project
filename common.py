from abc import ABC, abstractmethod
from spacecraft import Environment
import pygame
from pathlib import Path


class Settings:
    TIME_STEP_SIZE = 1/5

    SAVES_DIR = Path(__file__).parent / "saves"

    PPO_SAVE_DIR = SAVES_DIR / "ppo"
    PPO_LANDER_BEST = PPO_SAVE_DIR / "lander" / "best"
    PPO_LANDER_CHECKPOINT = PPO_SAVE_DIR / "lander" / "checkpoint"
    PPO_CONTROL_BEST = PPO_SAVE_DIR / "control" / "best"
    PPO_CONTROL_CHECKPOINT = PPO_SAVE_DIR / "control" / "checkpoint"

    DQN_SAVE_DIR = SAVES_DIR / "dqn"
    DQN_LANDER_BEST = DQN_SAVE_DIR / "lander" / "best"
    DQN_LANDER_CHECKPOINT = DQN_SAVE_DIR / "lander" / "checkpoint"
    DQN_CONTROL_BEST = DQN_SAVE_DIR / "control" / "best"
    DQN_CONTROL_CHECKPOINT = DQN_SAVE_DIR / "control" / "checkpoint"
    
    DDPG_SAVE_DIR = SAVES_DIR / "ddpg"
    DDPG_LANDER_BEST = DDPG_SAVE_DIR / "lander" / "best"
    DDPG_LANDER_CHECKPOINT = DDPG_SAVE_DIR / "lander" / "checkpoint"
    DDPG_CONTROL_BEST = DDPG_SAVE_DIR / "control" / "best"
    DDPG_CONTROL_CHECKPOINT = DDPG_SAVE_DIR / "control" / "checkpoint"

    SIMULATION_FPS = 30
    SIMULATION_FRAME_SIZE = (800, 800)
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

    def render(self, surface: pygame.Surface):
        """
        Render something to the window. Like state variables or personal thoughts.
        """
        pass

    def handle_event(self, event: pygame.event.Event):
        """
        Can be used to handle events like clicking or mouse presses. 
        """
        pass

    def reset(self):
        """
        If agent returns actions sequentially from a list, this could be used to reset the index that points to the current action to return.
        """
        pass
