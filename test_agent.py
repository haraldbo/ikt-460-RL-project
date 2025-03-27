from environment_renderer import Renderer
from spacecraft import Environment
import pygame
from common import Settings
from copy import copy
from common import Agent
from ppo import PPOLandingAgent
from stable_baselines3 import PPO


def test_agent(agent: Agent, init_env: Environment):
    clock = pygame.time.Clock()
    pygame.init()
    pygame.font.init()
    env = copy(init_env)
    window = pygame.display.set_mode(Settings.SIMULATION_FRAME_SIZE)
    pygame.display.set_caption("Spacecraft control")
    renderer = Renderer()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Exiting")
                exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    env = copy(init_env)

        render_img = renderer.render(env)

        window_img = pygame.transform.scale(render_img, size=window.get_size())

        window.blit(window_img, dest=(0, 0))

        action = agent.get_action(env)
        env.step(action)

        clock.tick(Settings.SIMULATION_FPS)

        pygame.display.update()


if __name__ == "__main__":
    landing_agent = PPOLandingAgent(
        PPO.load(Settings.PPO_LANDER_BEST / "best_model", device="cpu"))
    init_env = Environment(time_step_size=Settings.TIME_STEP_SIZE)
    init_env.angular_velocity = 0.1
    test_agent(landing_agent, init_env)
