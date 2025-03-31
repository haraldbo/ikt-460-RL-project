import numpy as np
from enum import IntEnum
from PIL import Image


class MapTile(IntEnum):
    AIR = 0
    SOLID = 1


class Map:
    def __init__(self, tile_map: np.ndarray, start_position: tuple[int, int]):
        self.start_position = start_position
        self.tile_map = tile_map
        self.height = tile_map.shape[0]
        self.width = tile_map.shape[1]


class Environment:

    STATE_FLIGHT = 1
    STATE_ENDED = 2

    MIN_GIMBAL_LEVEL = -5
    MAX_GIMBAL_LEVEL = 5

    MIN_THRUST_LEVEL = 0
    MAX_THRUST_LEVEL = 10

    def __init__(self, gravity=-9.81, time_step_size=1/30, map: Map = None):

        self.gravity = gravity
        self.time_step_size = time_step_size

        if map == None:
            # The default map:
            # - Open area
            map_tiles = np.full((600, 600), MapTile.AIR, dtype=np.uint8)
            # - Flat ground level
            for y in range(0, 10):
                for x in range(map_tiles.shape[1]):
                    map_tiles[y, x] = MapTile.SOLID

            start_position = (map_tiles.shape[0]//2, map_tiles.shape[1]//2)
            self.map = Map(map_tiles, start_position=start_position)

        # Spacecraft properties
        self.width = 48.0
        self.height = 64.0
        self.mass = 500
        self.moment = self.mass * 1/12 * (self.height ** 2 + self.width ** 2)
        self.d_engine_com = 20  # Distance from engine to center of mass
        self.max_engine_gimbal_angle = np.pi/8
        self.max_thrust = -2 * self.mass * gravity
        self.min_thrust = -0.7 * self.mass * gravity
        self.max_gimbal_level = self.MAX_GIMBAL_LEVEL

        self.action_gimbal_range = (-1, 1)
        self.action_thrust_range = (-1, 1)

        self.reset()

    def reset(self):
        self.steps = 0
        self.position = self.map.start_position
        self.velocity = (0.0, 0.0)
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.thrust_level = 0
        self.gimbal_level = 0
        self.state = self.STATE_FLIGHT

    def get_engine_absolute_location(self):
        dy = self.d_engine_com * np.sin(-np.pi/2 + self.angle)
        dx = self.d_engine_com * np.cos(-np.pi/2 + self.angle)
        ship_x, ship_y = self.position

        return (ship_x + dx, ship_y + dy)

    def get_engine_absolute_angle(self):
        return -np.pi/2 + self.get_engine_local_angle() + self.angle

    def get_engine_local_angle(self):
        return (self.gimbal_level / self.MAX_GIMBAL_LEVEL) * self.max_engine_gimbal_angle

    def _get_thrust_velocity_change(self):
        """
        Returns velocity change of center of mass (COM) and the angular velocity change around COM:
        - x_velocity_change, y_velocity_change, angular_velocity_change
        """
        theta = self.get_engine_absolute_angle()

        if self.thrust_level > 0:
            thrust = self.min_thrust + (self.thrust_level-1) * \
                (self.max_thrust - self.min_thrust) / self.MAX_THRUST_LEVEL
        else:
            thrust = 0

        J = thrust * self.time_step_size

        # velocity change of ship is in opposite direction of jet
        velocity_change = -J/self.mass

        torque = thrust * self.d_engine_com * \
            np.sin(self.get_engine_local_angle())
        angular_acceleration = torque/self.moment

        angular_velocity_change = angular_acceleration * self.time_step_size

        x_velocity_change = velocity_change * np.cos(theta)
        y_velocity_change = velocity_change * np.sin(theta)

        return x_velocity_change, y_velocity_change, angular_velocity_change

    def get_collision_vertices(self):
        x, y = self.position

        # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
        corners_neutral = np.array([
            [- self.width/2, self.height / 2],  # (left, top)
            [self.width/2,  self.height / 2],  # (right, top)
            [- self.width/2 + 5, -self.height / 2],  # (left, bottom)
            [self.width/2 - 5, -self.height / 2]  # (right, bottom)
        ]).T

        rotation_matrix = np.array([
            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ])

        return (rotation_matrix @ corners_neutral).T + [x, y]

    def check_collision(self):
        for x, y in self.get_collision_vertices():
            if y < 0 or x < 0:
                return True
            if y > self.map.height or x > self.map.width:
                return True
            if self.map.tile_map[int(y), int(x)] == MapTile.SOLID:
                return True
        return False

    def get_distance_to(self, x, y):
        return np.sqrt((self.position[0] - x) ** 2 + (self.position[1] - y) ** 2)

    def get_velocity(self):
        return np.sqrt((self.velocity[0]) ** 2 + (self.velocity[1]) ** 2)

    def flight_has_ended(self):
        return self.state == self.STATE_ENDED

    def has_lifted_off(self):
        return self.state in [self.STATE_FLIGHT, self.STATE_ENDED]

    def validate_action(self, gimbal_action, thrust_action):
        if gimbal_action < self.action_gimbal_range[0] or gimbal_action > self.action_gimbal_range[1]:
            raise ValueError(f"Invalid gimbal action: {gimbal_action}")

        if thrust_action < self.action_thrust_range[0] or thrust_action > self.action_thrust_range[1]:
            raise ValueError(f"Invalid thrust action: {thrust_action}")

    def _perform_action(self, action: tuple[float, float]):
        gimbal_action, thrust_action = action
        self.validate_action(gimbal_action, thrust_action)

        self.gimbal_level = max(-self.MAX_GIMBAL_LEVEL,
                                min(self.gimbal_level + gimbal_action, self.MAX_GIMBAL_LEVEL))
        self.thrust_level = max(self.MIN_THRUST_LEVEL, min(
            self.thrust_level + thrust_action, self.MAX_THRUST_LEVEL))

    def _get_y_velocity(self):
        y_velocity_gravity = self.gravity * self.time_step_size
        _, y_velocity_change_thrust, _ = self._get_thrust_velocity_change()
        return self.velocity[1] + y_velocity_gravity + y_velocity_change_thrust

    def _update_flight_variables(self):
        y_velocity_gravity = self.gravity * self.time_step_size

        x_velocity_change_thrust, y_velocity_change_thrust, angular_velocity_change_thrust = self._get_thrust_velocity_change()

        # Calculate change in velocity caused by thrust
        x_velocity = self.velocity[0] + x_velocity_change_thrust
        y_velocity = self.velocity[1] + \
            y_velocity_gravity + y_velocity_change_thrust

        angular_velocity = self.angular_velocity + angular_velocity_change_thrust

        # Update velocity, angular velocity, angle and position of vehicle
        avg_velo_x = (x_velocity + self.velocity[0])/2
        avg_velo_y = (y_velocity + self.velocity[1])/2
        updated_position = (self.position[0] + avg_velo_x * self.time_step_size,
                            self.position[1] + avg_velo_y * self.time_step_size)
        self.position = updated_position
        self.velocity = (x_velocity, y_velocity)
        avg_angle_velo = (self.angular_velocity + angular_velocity)/2
        self.angle = self.angle + avg_angle_velo * self.time_step_size
        self.angular_velocity = angular_velocity

    def step(self, action: tuple[float, float]):
        if self.state == self.STATE_FLIGHT:
            self._perform_action(action)
            self._update_flight_variables()
            has_collided = self.check_collision()
            self.steps += 1

            if has_collided:
                self.thrust_level = 0
                self.state = self.STATE_ENDED

            return
        elif self.state == self.STATE_ENDED:
            return
        else:
            raise ValueError(f"Unrecognized state {self.state}")
