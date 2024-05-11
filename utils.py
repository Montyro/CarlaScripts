# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse


import math
import numpy as np
import numpy as np
from math import radians, degrees

def euler_to_rotation_matrix(euler_angles):
    rx, ry, rz = np.radians(euler_angles)
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)

    rotation_matrix = np.array([
        [cy*cz, -cy*sz, sy],
        [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy],
        [sx*sz - cx*sy*cz, sx*cz + cx*sy*sz, cx*cy]
    ])

    return rotation_matrix

def rotation_matrix_to_euler(rotation_matrix):
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

def get_relative_rotation(parent_world_rotation, parent_relative_rotation, required_angle, vertical_offset):

    parent_world_rotation = np.radians(parent_world_rotation)
    parent_relative_rotation = np.radians(parent_relative_rotation)
    required_angle = np.radians(required_angle)
    vertical_offset = np.radians(vertical_offset)

    # Convert Euler angles to rotation matrices
    parent_world_matrix = euler_to_rotation_matrix(parent_world_rotation)
    parent_relative_matrix = euler_to_rotation_matrix(parent_relative_rotation)
    required_angle_matrix = euler_to_rotation_matrix(required_angle)
    vertical_offset_matrix = euler_to_rotation_matrix(vertical_offset)

    # Calculate the joint's world space rotation matrix
    joint_world_matrix = parent_world_matrix @ parent_relative_matrix

    # Apply the vertical offset to the required angle
    required_angle_matrix_offset = vertical_offset_matrix @ required_angle_matrix

    # Calculate the joint's relative rotation matrix
    joint_relative_matrix = np.linalg.inv(joint_world_matrix) @ required_angle_matrix_offset

    # Convert the joint's relative rotation matrix back to Euler angles
    joint_relative_rotation = rotation_matrix_to_euler(joint_relative_matrix)

    return np.degrees(joint_relative_rotation)
