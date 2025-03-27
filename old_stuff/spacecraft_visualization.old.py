from spacecraft import Environment
import numpy as np
import pygame
from pygame.font import Font
from pygame.transform import scale
from PIL import Image
import copy
from common import Agent


class ExplosionRenderer:

    def __init__(self):
        self.frame = 0
        self.max_frames = 100

    def reset(self):
        self.frame = 0

    def is_done(self):
        return self.frame >= self.max_frames

    def render(self, surface: pygame.Surface, e: Environment):
        if self.frame >= self.max_frames:
            return

        radius = 2 + np.sin(np.pi * (self.frame / self.max_frames)) * 100

        self.frame += 1
        for i in range(min(self.frame, 20)):
            color = (200 + np.random.randint(-40, 40),
                     150 + np.random.randint(-50, 50), 0)

            x = np.random.randint(
                e.position[0] - radius, e.position[0] + radius)

            y = 600 - \
                np.random.randint(
                    e.position[1] - radius, e.position[1] + radius)

            pygame.draw.circle(surface, color, center=(
                x, y), radius=np.random.randint(1, radius))

            if self.frame > 30:
                pygame.draw.line(surface, color, start_pos=(x, y), end_pos=(
                    x+np.random.randint(-radius, radius), y + np.random.randint(-radius, radius)))
                pygame.draw.arc(surface, color, start_angle=0, stop_angle=np.random.rand(), rect=(
                    (x, y), (x+np.random.randint(-radius, radius), y + np.random.randint(-radius, radius))))


class RocketJetRenderer:

    def render(self, surface: pygame.Surface, e: Environment):
        engine_angle = -np.pi/2 - e.get_engine_local_angle() + e.angle
        radius = max(14, 8 + e.thrust_level)
        for i in range(np.random.randint(5, 20)):
            flame_start = (e.position[0] + radius * np.cos(-np.pi/2 + e.angle),
                           600 - e.position[1] - radius * np.sin(-np.pi/2 + e.angle))
            flame_length = np.random.randint(10, 40) + e.thrust_level * 2
            flame_end = (flame_start[0] + np.cos(engine_angle) * flame_length,
                         flame_start[1] - np.sin(engine_angle) * flame_length
                         )
            color = (
                254 + np.random.randint(-1, 1),
                190 + np.random.randint(0, 40),
                180 + np.random.randint(0, 55)
            )
            src = (flame_start[0] + np.random.randint(-1, 1),
                   flame_start[1] + np.random.randint(-1, 1))

            spray = min(e.thrust_level, 3)
            dest = [flame_end[0] + np.random.randint(-spray, spray),
                    flame_end[1] + np.random.randint(-spray, spray)]

            if dest[1] > 600 - e.ground_line:
                dest[0] = flame_end[0] + np.random.randint(-2, 2)
                dest[1] = 600 - e.ground_line

            jet_width = int(1 + 3 * e.thrust_level/e.MAX_THRUST_LEVEL)
            pygame.draw.line(surface, color, src, dest, width=jet_width)


class SpacecraftRenderer:

    def __init__(self, render_collision_vertices):
        self.spacecraft_img = pygame.image.load("images/spacecraft.png")
        self.render_collision_vertices = render_collision_vertices

    def render(self, surface: pygame.Surface, e: Environment):
        rotated = pygame.transform.rotozoom(
            self.spacecraft_img, e.angle/(np.pi * 2) * 360, 0.5)
        surface.blit(source=rotated, dest=(
            e.position[0] - rotated.get_width()/2, (600 - e.position[1] - rotated.get_height()/2)))
        if self.render_collision_vertices:

            pygame.draw.circle(surface, (0, 255, 0), center=(
                e.position[0], 600 - e.position[1]), radius=1)


class EnvironmentRenderer:

    def render(self, surface: pygame.Surface, environment: Environment):
        surface.fill((10, 0, 20))  # Sky color

        surface.fill(
            (160, 140, 100),  # Ground color
            (0, surface.get_height() - 10,
             surface.get_width(), surface.get_height())
        )

        r = 32
        landing_area = (environment.map.width//2, 10)
        pygame.draw.circle(
            surface, (0, 255, 0), (landing_area[0] + r, surface.get_height() - landing_area[1]), radius=1)
        pygame.draw.circle(
            surface, (0, 255, 0), (landing_area[0] - r, surface.get_height() - landing_area[1]), radius=1)


class InfoRenderer:

    def __init__(self, font: Font):
        self.font = font

    def render(self, surface: pygame.Surface, info: dict[str, str]):
        y_loc = 0
        for k, v in info.items():
            text = f"{k}: {v}"
            font_surface = self.font.render(text, False, (255, 255, 255))
            surface.blit(font_surface, dest=(450, 20 + y_loc))
            y_loc += self.font.size(text)[1]


class DustRenderer:

    def __init__(self):
        # Using a circular buffer
        self.history = self._init_history()
        self.frame = 0

    def _init_history(self):
        return [((0, 0), 0) for i in range(30)]

    def reset(self):
        self.frame = 0
        self.history = self._init_history()

    def _render_history(self, surface: pygame.Surface, environment: Environment):
        for k in range(len(self.history)):
            (x, y), intensity = self.history[(
                self.frame - 1 - k) % len(self.history)]

            intensity *= (0.8 ** k)

            intensity = min(intensity, 14)

            y_start = np.random.randint(0, 3)
            y_end = np.random.randint(y_start + 1, y_start + 4)
            if np.random.rand() > 0.5:  # right side
                x_start = np.random.randint(2, 5)
                x_end = np.random.randint(
                    x_start + 3, x_start + max(6, intensity))
            else:  # left side
                x_start = np.random.randint(-5, 2)
                x_end = np.random.randint(
                    x_start - max(6, intensity), x_start - 3)

            color = (178 + np.random.randint(-10, 10), 163 +
                     np.random.randint(-10, 10), 132 + np.random.randint(-10, 10))

            pygame.draw.line(surface, color=color, start_pos=(
                x + x_start, 600 - y - y_start), end_pos=(x_end + x, 600 - y - y_end), width=int(min(intensity, 3)))

    def is_done(self):
        for ((_, _), intensity) in self.history:
            if intensity != 0:
                return False
        return True

    def render(self, surface: pygame.Surface, environment: Environment):

        engine_abs_angle = environment.get_engine_absolute_angle()
        if np.fabs(np.sin(engine_abs_angle)) < 0.1 or environment.thrust_level == 0 or np.cos(environment.angle) < 0:
            dust_x = 0
            dust_y = 0
            intensity = 0
        else:
            x, y = environment.get_engine_absolute_location()
            a = (y - environment.ground_line) / \
                np.sin(engine_abs_angle) * np.cos(engine_abs_angle)

            dust_x = x - a
            dust_y = environment.ground_line
            intensity = (10_000 * environment.thrust_level) / \
                (np.linalg.norm((dust_x - x, dust_y - y)) ** 2)

        self.history[self.frame % len(self.history)] = (
            (dust_x, dust_y), intensity)
        self.frame += 1
        self._render_history(surface, environment)


def save_animation(frames: list[pygame.Surface]):
    images = []
    for i, frame in enumerate(frames):
        if i % 3 == 0:
            continue
        frame_bytes = pygame.image.tobytes(frame, "RGB")
        img = Image.frombytes(
            mode="RGB", size=frame.get_size(), data=frame_bytes)
        images.append(img.resize(
            (400, 400), resample=Image.Resampling.BILINEAR))

    for i in range(10):
        images.append(images[-1])

    img = images[0]

    img.save(fp="animation.gif", format='GIF', append_images=images[1:],
             save_all=True, duration=70, loop=0, optimize=True)


def spacecraft_has_exploded(environment: Environment):
    is_far_from_landing_site = 10 > 100
    too_high_velocity = np.fabs(
        environment.angular_velocity) > 0.5 or environment.get_velocity() > 10
    too_high_angle = np.fabs(environment.angle) > 0.1
    return environment.flight_has_ended() and (is_far_from_landing_site or too_high_angle or too_high_velocity)


def start_visualization(initial_environment: Environment, agent: Agent, fps=100, save_animation_frames=False):

    environment = copy.copy(initial_environment)
    clock = pygame.time.Clock()
    pygame.init()
    pygame.font.init()

    font: Font = pygame.font.SysFont("", size=16)
    window = pygame.display.set_mode((600, 600))
    world_surface = pygame.Surface(window.get_size())

    pygame.display.set_caption("Spacecraft control")

    # jet_renderer = RocketJetRenderer()
    spacecraft_renderer = SpacecraftRenderer(render_collision_vertices=False)
    background_renderer = EnvironmentRenderer()
    info_renderer = InfoRenderer(font)
    explosion_renderer = ExplosionRenderer()
    # dust_renderer = DustRenderer()

    animation_frames = []
    while True:

        action = None
        for event in pygame.event.get():

            agent.handle_event(event)

            if event.type == pygame.QUIT:
                print("Exiting")
                exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Reseting")
                    environment = copy.copy(initial_environment)
                    explosion_renderer.reset()
                    # dust_renderer.reset()
                    animation_frames.clear()
                    agent.reset()
                    continue
                if event.key == pygame.K_s:
                    if save_animation_frames:
                        print("Saving animation")
                        save_animation(animation_frames)
                    else:
                        print("Animation is not enabled")

        action = agent.get_action(environment)

        environment.step(action)

        background_renderer.render(world_surface, environment)
        has_exploded = spacecraft_has_exploded(environment)
        if has_exploded:
            explosion_renderer.render(world_surface, environment)
        else:
            # if environment.thrust_level > 0:
            # jet_renderer.render(world_surface, environment)
            spacecraft_renderer.render(world_surface, environment)
            # dust_renderer.render(world_surface, environment)

        agent.render(world_surface)

        if save_animation_frames:
            animation_frames.append(world_surface.copy())

        crop_size = 200
        x = min(
            max(environment.position[0]-crop_size//2, 0), window.get_width() - crop_size)
        y = min(max((window.get_height(
        ) - environment.position[1])-crop_size//2, 0), window.get_height() - crop_size)

        canvas = world_surface.subsurface(x, y,  crop_size, crop_size)

        canvas = scale(canvas, window.get_size())

        info_renderer.render(canvas, {
            "Position": (round(environment.position[0], 2), round(environment.position[1], 2)),
            "Velocity": (round(environment.velocity[0], 2), round(environment.velocity[1], 2)),
            "Absolute velocity": (round(environment.get_velocity(), 2)),
            "Angle": round(environment.angle, 3),
            "Angular velocity": round(environment.angular_velocity, 3),
            "Thrust level": environment.thrust_level,
            "Gimbal": environment.gimbal_level,
            "Steps": environment.steps,
            "Flight ended": environment.flight_has_ended(),
            "FPS": round(clock.get_fps(), 0),
        })

        window.blit(canvas, (0, 0))
        pygame.display.update()
        clock.tick(fps)
