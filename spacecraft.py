import numpy as np


class Environment:
    WORLD_SIZE = 600

    STATE_LAUNCH = 0
    STATE_IN_FLIGHT = 1
    STATE_ENDED = 2

    MIN_GIMBAL_LEVEL = -5
    MAX_GIMBAL_LEVEL = 5

    MIN_THRUST_LEVEL = 0
    MAX_THRUST_LEVEL = 10

    def __init__(self, gravity=-9.81, time_step_size=1/30):
        self.gravity = gravity
        self.width = 48.0
        self.height = 64.0
        self.mass = 500
        self.ground_line = 50
        self.landing_area = (self.WORLD_SIZE//2, self.ground_line)
        self.launch_pad = self.landing_area
        self.moment = self.mass * 1/12 * (self.height ** 2 + self.width ** 2)

        # Distance from engine to center of mass
        self.d_engine_com = 20

        self.time_step_size = time_step_size

        self.max_engine_gimbal_angle = np.pi/8

        # Works in the opposite direction of gravitational force
        self.max_thrust = -2 * self.mass * gravity
        self.min_thrust = -0.7 * self.mass * gravity

        self.max_thrust_level = self.MAX_THRUST_LEVEL

        # Number of angle settings for each side
        self.max_gimbal_level = self.MAX_GIMBAL_LEVEL

        self.action_space = []
        for gimbal_action in range(-1, 2):
            for thrust_action in range(-1, 2):
                self.action_space.append((gimbal_action, thrust_action))

        self.reset()

    def reset(self):
        self.steps = 0
        self.position = (self.launch_pad[0],
                         self.launch_pad[1] + self.height//4)
        self.velocity = (0.0, 0.0)

        # Positive direction is counterclockwise
        self.angle = 0.0
        self.angular_velocity = 0.0

        self.thrust_level = 0  # from 0 to 6
        self.gimbal_level = 0  # from -6 to 6

        self.state = self.STATE_LAUNCH

        self._update_collision_variables()

    def get_engine_absolute_location(self):
        dy = self.d_engine_com * np.sin(-np.pi/2 + self.angle)
        dx = self.d_engine_com * np.cos(-np.pi/2 + self.angle)
        ship_x, ship_y = self.position

        return (ship_x + dx, ship_y + dy)

    def get_engine_absolute_angle(self):
        return -np.pi/2 + self.get_engine_local_angle() + self.angle

    def get_engine_local_angle(self):
        return (self.gimbal_level / self.max_gimbal_level) * self.max_engine_gimbal_angle

    def _get_thrust_velocity_change(self):
        theta = self.get_engine_absolute_angle()

        if self.thrust_level > 0:
            thrust = self.min_thrust + (self.thrust_level-1) * \
                (self.max_thrust - self.min_thrust) / (self.max_thrust_level-1)
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

    def _update_collision_variables(self):
        x, y = self.position
        has_collided = False

        # Check if spacecraft collided with groud level
        if y - self.height / 4 < self.ground_line:
            has_collided = True

        if y > self.WORLD_SIZE or x > self.WORLD_SIZE or x < 0:
            has_collided = True

        self.has_collided = has_collided

    def get_distance_to_landing_site(self):
        return np.sqrt((self.position[0] - self.landing_area[0]) ** 2 + (self.position[1] - self.landing_area[1]) ** 2)

    def get_distance_to(self, x, y):
        return np.sqrt((self.position[0] - x) ** 2 + (self.position[1] - y) ** 2)

    def get_velocity(self):
        return np.sqrt((self.velocity[0]) ** 2 + (self.velocity[1]) ** 2)

    def flight_has_ended(self):
        return self.state == self.STATE_ENDED

    def has_lifted_off(self):
        return self.state in [self.STATE_IN_FLIGHT, self.STATE_ENDED]

    def _perform_action(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}")

        self.gimbal_level = max(-self.max_gimbal_level, min(
            self.gimbal_level + action[0], self.max_gimbal_level))

        min_thrust_level = 1 if self.has_lifted_off() else 0
        self.thrust_level = max(
            min_thrust_level, min(self.thrust_level + action[1], self.max_thrust_level))

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
        self._update_collision_variables()

    def step(self, action):
        if self.state == self.STATE_LAUNCH:
            self._perform_action(action)
            self.steps += 1

            if self._get_y_velocity() <= 0:
                return self.state == self.STATE_ENDED

            self.state = self.STATE_IN_FLIGHT
            self._update_flight_variables()
            return self.state == self.STATE_ENDED

        elif self.state == self.STATE_IN_FLIGHT:
            self._perform_action(action)

            self._update_flight_variables()

            if self.has_collided:
                self.thrust_level = 0
                self.state = self.STATE_ENDED

            self.steps += 1
            return self.state == self.STATE_ENDED

        elif self.state == self.STATE_ENDED:
            return True
        else:
            raise ValueError(f"Unrecognized state {self.state}")
