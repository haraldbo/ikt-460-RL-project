from spacecraft import Environment, Map, MapTile
from pygame import Surface
import pygame
import numpy as np
from common import Settings
from collections import deque


class Colors:
    SPACE = (10, 0, 20)
    SOLID = (160, 140, 100)


class Renderer:

    def __init__(self):
        self.map_image = None
        self.spacecraft_img = pygame.image.load("images/spacecraft.png")
        self.render_image = None
        self.top = 0
        self.left = 0
        self.font = pygame.font.SysFont("", size=16)
        self.dust = deque(maxlen=20)

    def _tile_to_color(self, tile: MapTile):
        if tile == MapTile.AIR:
            return Colors.SPACE
        else:
            return Colors.SOLID

    def _map_to_image(self, map: Map):
        map_img = Surface((map.width, map.height))

        for y in range(map.height):
            for x in range(map.width):
                map_img.set_at((x, y), self._tile_to_color(
                    map.tile_map[map.height - y - 1, x]))

        if Settings.RENDERING_SPACECRAFT_DEBUGGING:
            pygame.draw.circle(map_img, color=(0, 255, 0), center=(
                map_img.get_width()//2, map_img.get_height() - 10), radius=3)

        n_stars = 20
        for i in range(n_stars):
            x = np.random.randint(0, map.width)
            y = np.random.randint(0, map.height)
            if map.tile_map[map.height - y - 1, x] == MapTile.AIR:
                pygame.draw.circle(map_img, color=(255, 255
                                                   - np.random.randint(0, 55), 255),
                                   center=(x, y), radius=np.random.randint(1, 3), width=np.random.randint(1, 3))

        return map_img

    def _update_viewport_left_top(self, env: Environment):
        x, y = env.position
        y_screen = env.map.height - y

        viewport_width, viewport_height = Settings.RENDERING_VIEWPORT_SIZE

        self.left = max(0, min(x - viewport_width//2,
                               env.map.width - viewport_width))
        self.top = max(0, min(y_screen - viewport_height //
                              2, env.map.height - viewport_height))

    def _render_map(self, env: Environment):
        if not self.map_image:
            self.map_image = self._map_to_image(env.map)

        self.render_image.blit(self.map_image, dest=(0, 0), area=(
            self.left, self.top, *Settings.RENDERING_VIEWPORT_SIZE))

    def _render_spacecraft(self, env: Environment):
        spacecraft_rotated = pygame.transform.rotozoom(
            self.spacecraft_img, env.angle/(np.pi * 2) * 360, 1)
        x, y = env.position

        x_dest = x - spacecraft_rotated.get_width() / 2 - self.left
        y_dest = env.map.height - y - spacecraft_rotated.get_height() / 2 - self.top

        self.render_image.blit(source=spacecraft_rotated,
                               dest=(x_dest, y_dest))

        if Settings.RENDERING_SPACECRAFT_DEBUGGING:
            # Center
            pygame.draw.circle(self.render_image, color=(
                0, 255, 0), center=(x - self.left, env.map.height - y - self.top), radius=2)

            # Collision vertices
            for x, y in env.get_collision_vertices():
                pygame.draw.circle(self.render_image, color=(
                    0, 0, 255), center=(x - self.left, env.map.height - y - self.top), radius=1)

    def _render_landing_area(self, env: Environment):
        landing_area = (env.map.width//2, 10)
        x, y = landing_area
        x_dest = x - self.left
        y_dest = env.map.height - y - self.top

        radius = 50

        pole_height = 30
        flag_height = 10
        flag_width = 7

        for r in [-radius, radius]:
            pole_start = (x_dest + r, y_dest)
            pole_end = (x_dest + r, y_dest - pole_height)
            pygame.draw.line(self.render_image, color=(
                255, 255, 255), start_pos=pole_start, end_pos=pole_end)

            flag_points = [
                (pole_end[0], pole_end[1]),
                (pole_end[0], pole_end[1] + flag_height),
                (pole_end[0] + flag_width, pole_end[1] + flag_height//2)
            ]
            pygame.draw.polygon(self.render_image, color=(
                200, 200, 0), points=flag_points)

    def _add_dust(self, env: Environment):
        engine_angle = -np.pi/2 - env.get_engine_local_angle() + env.angle
        engine_location = (env.position[0] + np.cos(-np.pi/2 + env.angle) * env.d_engine_com,
                           env.position[1] + np.sin(-np.pi/2 + env.angle) * env.d_engine_com)

        dx = np.cos(engine_angle)
        dy = np.sin(engine_angle)

        if dy >= -0.001:
            return

        m = (10 - engine_location[1])/dy

        dust_x = env.position[0] + m * dx
        dust_location = (dust_x, 10)

        intensity = np.fabs((100 * env.thrust_level) /
                            (np.fabs(10 - engine_location[1]) + 1))

        self.dust.appendleft((intensity, dust_location))

    def _render_dust(self, env: Environment):
        self._add_dust(env)

        if len(self.dust) == 0:
            return

        dust_intensity, dust_location = self.dust.pop()
        x, y = dust_location
        x_dest = x - self.left
        y_dest = env.map.height - y - self.top

        for y in range(int(dust_intensity * 2)):
            dx1 = np.random.randint(-10, 10)
            dx2 = np.sign(dx1) * np.random.randint(0, 1 + int(dust_intensity))
            dy = np.sqrt(np.random.randint(0, 1 + int(dust_intensity)))

            line_width = 1 + np.random.randint(0, 1 + int(dust_intensity/15))

            pygame.draw.line(self.render_image, color=(150 + np.random.randint(-20, 0), 130 + np.random.randint(-20, 0), 90 + np.random.randint(-20, 0)),
                             start_pos=(x_dest+dx1, y_dest),
                             end_pos=(x_dest+dx1+dx2, y_dest - dy), width=line_width)

        if Settings.RENDERING_SPACECRAFT_DEBUGGING:
            x, y = (env.position[0] + np.cos(-np.pi/2 + env.angle) * env.d_engine_com,
                    env.position[1] + np.sin(-np.pi/2 + env.angle) * env.d_engine_com)
            x_dest = x - self.left
            y_dest = env.map.height - y - self.top

            pygame.draw.circle(self.render_image, color=(
                255, 0, 0), center=(x_dest, y_dest), radius=2)

    def _render_jet(self, env: Environment):
        # TODO: Check if has collided with solid
        if env.thrust_level == 0:
            return

        engine_angle = -np.pi/2 - env.get_engine_local_angle() + env.angle
        radius = max(14, 8 + env.thrust_level)

        x, y = env.position
        x_dest = x - self.left
        y_dest = env.map.height - y - self.top

        y_max_jet = env.map.height - 10 - self.top

        for i in range(np.random.randint(5, 20)):
            flame_start = (x_dest + radius * np.cos(-np.pi/2 + env.angle),
                           y_dest - radius * np.sin(-np.pi/2 + env.angle))
            flame_length = int(np.random.randint(
                10, 40) + env.thrust_level * 2)
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

            spray = min(np.ceil(env.thrust_level), 3)
            dest = [flame_end[0] + np.random.randint(-spray, spray),
                    min(flame_end[1] + np.random.randint(-spray, spray), y_max_jet)]

            jet_width = int(1 + 3 * env.thrust_level/env.MAX_THRUST_LEVEL)
            pygame.draw.line(self.render_image, color,
                             src, dest, width=jet_width)

    def render_spacecraft_information(self, environment: Environment, surface: Surface, extras):
        """
        Renders spacecraft information onto the surface
        """

        info = {
            "Position": (round(environment.position[0], 3), round(environment.position[1], 3)),
            "Velocity": (round(environment.velocity[0], 3), round(environment.velocity[1], 3)),
            "Angle": round(environment.angle, 3),
            "Angular Velocity":  round(environment.angular_velocity, 3),
            "Thrust":  round(environment.thrust_level, 3),
            "Gimbal":  round(environment.gimbal_level, 3),
            **extras
        }

        y_loc = 0
        for k, v in info.items():

            text = f"{k}: {v}"
            font_surface = self.font.render(text, False, (255, 255, 255))
            surface.blit(font_surface, dest=(20, 20 + y_loc))
            y_loc += self.font.size(text)[1]

    def render(self, environment: Environment):
        """
        renders the environment and the spacecraft
        """
        if self.render_image == None:
            self.render_image = Surface(Settings.RENDERING_VIEWPORT_SIZE)

        self._update_viewport_left_top(environment)
        self._render_map(environment)
        self._render_jet(environment)
        self._render_spacecraft(environment)
        self._render_landing_area(environment)
        self._render_dust(environment)

        return self.render_image
