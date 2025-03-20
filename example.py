from spacecraft_visualization import start_visualization, Agent
from spacecraft import Environment
import pygame


class HumanPlayerAgent(Agent):

    def __init__(self):
        self.next_action = (0, 0)

    def get_action(self, environment: Environment):
        action = (
            max(-1, min(self.next_action[0], 1)),
            max(-1, min(self.next_action[1], 1))
        )

        self.next_action = (0, 0)
        return action

    def render(self, window: pygame.Surface):
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            gimbal_action = 0
            thrust_action = 0

            if event.key == pygame.K_UP:
                thrust_action = 1
            if event.key == pygame.K_DOWN:
                thrust_action = -1
            if event.key == pygame.K_LEFT:
                gimbal_action = 1
            if event.key == pygame.K_RIGHT:
                gimbal_action = -1

            # Done to allow actions such as (1, -1):
            self.next_action = (
                self.next_action[0] + gimbal_action,
                self.next_action[1] + thrust_action
            )

    def reset(self):
        pass


environment = Environment(time_step_size=1)
agent = HumanPlayerAgent()
start_visualization(environment, fps=10, agent=agent,
                    save_animation_frames=True)
