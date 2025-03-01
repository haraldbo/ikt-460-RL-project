from spacecraft_visualization import start_visualization, Agent
from spacecraft import Environment
import pygame


class HumanPlayerAgent(Agent):

    def __init__(self):
        self.next_action = None

    def get_action(self, environment: Environment):
        if self.next_action != None:
            action = self.next_action
            self.next_action = None
            return action

        return (0, 0)

    def render(self, window: pygame.Surface):
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.next_action = (0, 1)
            if event.key == pygame.K_LEFT:
                self.next_action = (1, 0)
            if event.key == pygame.K_RIGHT:
                self.next_action = (-1, 0)
            if event.key == pygame.K_DOWN:
                self.next_action = (0, -1)

    def reset(self):
        pass


environment = Environment(time_step_size=1/10)
agent = HumanPlayerAgent()
start_visualization(environment, fps=30, agent=agent,
                    save_animation_frames=True)
