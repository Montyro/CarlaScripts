import math
import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation

def calculate_relative_rotation(parent_world_rotation, desired_rotation):
    # Convert the parent's world rotation from Euler angles (degrees) to a rotation matrix
    parent_rotation = Rotation.from_euler('xyz', parent_world_rotation, degrees=True)
    parent_rotation_matrix = parent_rotation.as_matrix()

    # Convert the desired rotation from Euler angles (degrees) to a rotation matrix
    desired_rotation = Rotation.from_euler('xyz', desired_rotation, degrees=True)
    desired_rotation_matrix = desired_rotation.as_matrix()

    # Calculate the relative rotation by multiplying the inverse of the parent's rotation matrix
    # with the desired rotation matrix
    relative_rotation_matrix = np.linalg.inv(parent_rotation_matrix) @ desired_rotation_matrix

    # Convert the relative rotation matrix back to Euler angles (degrees)
    relative_rotation = Rotation.from_matrix(relative_rotation_matrix).as_euler('xyz', degrees=True)

    return relative_rotation

def euler_to_rotation_matrix(euler_angles):
    # Convert Euler angles (degrees) to radians
    angles_rad = np.radians(euler_angles)

    # Extract individual angles
    rx, ry, rz = angles_rad

    # Calculate the rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])

    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])

    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])

    # Combine the rotation matrices (assuming the order is x, y, z)
    rotation_matrix = Rz @ Ry @ Rx

    return rotation_matrix

def rotation_matrix_to_euler(rotation_matrix):
    # Extract the Euler angles (degrees) from the rotation matrix
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    # Convert angles from radians to degrees
    euler_angles = np.degrees([x, y, z])

    return euler_angles