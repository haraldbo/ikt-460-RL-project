from environment_renderer import Renderer
from spacecraft import Environment
import pygame
from common import Settings
from copy import copy
from common import Agent
from ppo_sb3 import PPOLandingAgent, PPOHoveringAgent
from ddpg_sb3 import DDPGLandingAgent
from td3_sb3 import TD3LandingAgent
from stable_baselines3 import PPO, DQN


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

        window_img = pygame.transform.scale(render_img, window.get_size())

        if Settings.RENDER_SPACECRAFT_INFORMATION:
            renderer.render_spacecraft_information(env, window_img)

        window.blit(window_img, dest=(0, 0))

        action = agent.get_action(env)
        env.step(action)

        clock.tick(Settings.SIMULATION_FPS)

        pygame.display.update()


if __name__ == "__main__":
    init_env = Environment(time_step_size=Settings.TIME_STEP_SIZE)
    init_env.position = (init_env.map.width//2+0, 200)
    landing_agent = PPOLandingAgent(landing_area=(init_env.map.width//2, 10))
    landing_agent = TD3LandingAgent(landing_area=(init_env.map.width//2, 10))
    # landing_agent = DDPGLandingAgent()
    # hovering_agent = PPOHoveringAgent(hovering_point=hover_point)
    
    init_env.angular_velocity = 0.0
    #test_agent(hovering_agent, init_env)
    test_agent(landing_agent, init_env)
