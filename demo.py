from environment_renderer import Renderer
from spacecraft import Environment
import pygame
from common import Settings
from copy import copy
from ppo import LandingAgent as PPOLandingAgent
from sac import LandingAgent as SACLandingAgent
from ddpg import LandingAgent as DDPGLandingAgent
from gyms import LandingSpacecraftGym
import numpy as np


class Mode:
    HOVERING = "hovering"
    LANDING = "landing"


class KEY_BINDINGS:
    LANDING = pygame.K_l
    HOVERING = pygame.K_h
    RIGHT = pygame.K_RIGHT
    DOWN = pygame.K_DOWN
    LEFT = pygame.K_LEFT
    UP = pygame.K_UP


class LandingAgent:

    def __init__(self):
        pass

    def get_action(self, env, target):
        pass


class RandomActionAgent(LandingAgent):

    def __init__(self):
        pass

    def get_action(self, env, target):
        return [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]


class HumanControlledLandingAgent(LandingAgent):

    def __init__(self):
        self.gimbal_action = 0
        self.thrust_action = 0
        super().__init__()

    def register_key_down(self, key):
        pass

    def register_key_up(self, key):
        if key == pygame.K_UP:
            self.thrust_action = 1

        if key == pygame.K_DOWN:
            self.thrust_action = -1

        if key == pygame.K_LEFT:
            self.gimbal_action = 1

        if key == pygame.K_RIGHT:
            self.gimbal_action = -1

    def get_action(self, env, target):
        action = [self.gimbal_action, self.thrust_action]
        self.gimbal_action = 0
        self.thrust_action = 0
        return action


def test_landing_agent(landing_agent: LandingAgent, landing_gym: LandingSpacecraftGym):
    clock = pygame.time.Clock()
    pygame.init()
    pygame.font.init()
    window = pygame.display.set_mode(Settings.SIMULATION_FRAME_SIZE)
    pygame.display.set_caption("Spacecraft control")
    renderer = Renderer()
    done = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Exiting")
                exit(0)
            if event.type == pygame.KEYDOWN:
                if type(landing_agent) == HumanControlledLandingAgent:
                    landing_agent.register_key_down(event.key)
                if event.key == pygame.K_SPACE:
                    done = False
                    landing_gym.reset()
            elif event.type == pygame.KEYUP:
                if type(landing_agent) == HumanControlledLandingAgent:
                    landing_agent.register_key_up(event.key)

        render_img = renderer.render(landing_gym.env)

        window_img = pygame.transform.scale(render_img, window.get_size())

        if Settings.RENDER_SPACECRAFT_INFORMATION:
            renderer.render_spacecraft_information(
                landing_gym.env,
                window_img,
                extras={}
            )

        window.blit(window_img, dest=(0, 0))

        if not done:
            action = landing_agent.get_action(
                landing_gym.env, landing_gym.landing_area)

            obs, reward, flight_ended, truncated, info = landing_gym.step(
                action)

            done = truncated or flight_ended

        clock.tick(Settings.SIMULATION_FPS)

        pygame.display.update()


if __name__ == "__main__":
    landing_gym = LandingSpacecraftGym(
        discrete_actions=False, relaxed_constraints=True)

    landing_agent = HumanControlledLandingAgent()
    landing_agent = PPOLandingAgent()
    landing_agent = DDPGLandingAgent()
    landing_agent = RandomActionAgent()
    landing_agent = SACLandingAgent()

    test_landing_agent(landing_agent, landing_gym)
