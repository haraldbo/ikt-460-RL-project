import matplotlib.pyplot as plt
import numpy as np


def collision_handling():
    height = 10
    width = 10
    x = 0
    y = 0
    angle = np.pi
    corners_neutral = np.array([
        [y + height / 2, x - width/2],  # (top, left)
        [y + height / 2, x + width/2],  # (top, right)
        [y - height / 2, x - width/2],  # (bottom, left)
        [y - height / 2, x + width/2]  # (bottom, right)
    ]).T

    print(corners_neutral)

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    corners_rotated = rotation_matrix @ corners_neutral

    print(corners_rotated)


collision_handling()
