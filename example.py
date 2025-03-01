from spacecraft_visualization import start_visualization, Agent
from spacecraft import Environment, ACTION_DO_NOTHING, ACTION_INCREASE_THRUST, ACTION_GIMBAL_LEFT, ACTION_GIMBAL_RIGHT, ACTION_DECREASE_THRUST
import pygame


class HumanPlayerAgent(Agent):

    def __init__(self):
        self.next_action = None

    def get_action(self, environment: Environment):
        if self.next_action != None:
            action = self.next_action
            self.next_action = None
            return action

        return ACTION_DO_NOTHING

    def render(self, window: pygame.Surface):
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.next_action = ACTION_INCREASE_THRUST
            if event.key == pygame.K_LEFT:
                self.next_action = ACTION_GIMBAL_LEFT
            if event.key == pygame.K_RIGHT:
                self.next_action = ACTION_GIMBAL_RIGHT
            if event.key == pygame.K_DOWN:
                self.next_action = ACTION_DECREASE_THRUST

    def reset(self):
        pass


environment = Environment(time_step_size=1/10)
agent = HumanPlayerAgent()
start_visualization(environment, fps=30, agent=agent,
                    save_animation_frames=True)
