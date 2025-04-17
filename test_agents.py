from environment_renderer import Renderer
from spacecraft import Environment
import pygame
from common import Settings
from copy import copy
from ppo import LandingAgent as PPOLandingAgent
from sac import LandingAgent as SACLandingAgent
from ddpg import LandingAgent as DDPGLandingAgent


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


class SpacecraftCommander:

    def __init__(self, hovering_agent, landing_agent, hovering_point, landing_point, mode=Mode.HOVERING):
        self.mode = mode
        self.landing_agent = landing_agent
        self.hovering_agent = hovering_agent
        self.init_hovering_point = hovering_point
        self.init_landing_point = landing_point
        self.key_down_map = {
            KEY_BINDINGS.LANDING: False,
            KEY_BINDINGS.HOVERING: False,
            KEY_BINDINGS.RIGHT: False,
            KEY_BINDINGS.DOWN: False,
            KEY_BINDINGS.LEFT: False,
            KEY_BINDINGS.UP: False,
        }
        self.point_change_amount = 5
        self.reset()

    def reset(self):
        self.landing_point = self.init_landing_point
        self.hovering_point = self.init_hovering_point
        self.mode = Mode.HOVERING

    def set_landing_mode(self):
        self.mode = Mode.LANDING

    def set_hovering_mode(self):
        self.mode = Mode.HOVERING

    def update_key_map(self, key, down):
        self.key_down_map[key] = down

    def tick(self):
        if self.mode != Mode.HOVERING and self.key_down_map[KEY_BINDINGS.HOVERING]:
            self.set_hovering_mode()
        elif self.mode != Mode.LANDING and self.key_down_map[KEY_BINDINGS.LANDING]:
            self.set_landing_mode()

        if self.mode == Mode.HOVERING:
            if self.key_down_map[KEY_BINDINGS.LEFT]:
                self.hovering_point = (
                    self.hovering_point[0] - self.point_change_amount, self.hovering_point[1])
            if self.key_down_map[KEY_BINDINGS.RIGHT]:
                self.hovering_point = (
                    self.hovering_point[0] + self.point_change_amount, self.hovering_point[1])
            if self.key_down_map[KEY_BINDINGS.UP]:
                self.hovering_point = (
                    self.hovering_point[0], self.hovering_point[1] + self.point_change_amount)
            if self.key_down_map[KEY_BINDINGS.DOWN]:
                self.hovering_point = (
                    self.hovering_point[0], self.hovering_point[1] - self.point_change_amount)

    def get_action(self, environment: Environment):
        if self.mode == Mode.HOVERING:
            return self.hovering_agent.get_action(environment, self.hovering_point)
        else:
            return self.landing_agent.get_action(environment, self.landing_point)


def test_flight(commander: SpacecraftCommander, init_env: Environment):
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
                    commander.reset()
                    env = copy(init_env)
                elif event.key == pygame.K_l:
                    commander.set_landing_mode()

                commander.update_key_map(event.key, down=True)
            if event.type == pygame.KEYUP:
                commander.update_key_map(event.key, down=False)

        commander.tick()
        render_img = renderer.render(env)

        window_img = pygame.transform.scale(render_img, window.get_size())

        if Settings.RENDER_SPACECRAFT_INFORMATION:
            renderer.render_spacecraft_information(env, window_img)

        window.blit(window_img, dest=(0, 0))

        action = commander.get_action(env)
        env.step(action)

        clock.tick(Settings.SIMULATION_FPS)

        pygame.display.update()


if __name__ == "__main__":
    init_env = Environment()
    init_env.position = (init_env.map.width//2, init_env.map.height-50)

    spacecraft_commander = SpacecraftCommander(
        landing_agent=SACLandingAgent(),
        hovering_agent=PPOLandingAgent(),
        landing_point=(init_env.map.width//2, 10),
        hovering_point=(init_env.map.width//2, init_env.map.height//2)
    )

    spacecraft_commander.set_landing_mode()

    test_flight(spacecraft_commander, init_env)
