#!/usr/bin/env python
# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import math
import argparse
import copy
import time
from multiprocessing import Pool
from PIL import Image
from utils import get_relative_rotation
from maths import calculate_relative_rotation
try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self
    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
    # ---------------

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_as_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # make the array writeable doing a deep copy
    array2 = copy.deepcopy(array)
    return array2

def draw_image(surface, array, blend=False):
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def get_screen_points(camera, K, image_w, image_h, points3d):
    
    # get 4x4 matrix to transform points from world to camera coordinates
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # build the points array in numpy format as (x, y, z, 1) to be operable with a 4x4 matrix
    points_temp = []
    for p in points3d:
        points_temp += [p.x, p.y, p.z, 1]
    points = np.array(points_temp).reshape(-1, 4).T
    
    # convert world points to camera space
    points_camera = np.dot(world_2_camera, points)
    
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    points = np.array([
        points_camera[1],
        points_camera[2] * -1,
        points_camera[0]])
    
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, points)

    # normalize the values and transpose
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]]).T

    return points_2d

def draw_points_on_buffer(buffer, image_w, image_h, points_2d, color, size=4):
    half = int(size / 2)
    # draw each point
    for p in points_2d:
        x = int(p[0])
        y = int(p[1])
        for j in range(y - half, y + half):
            if (j >=0 and j <image_h):
                for i in range(x - half, x + half):
                    if (i >=0 and i <image_w):
                        buffer[j][i][0] = color[0]
                        buffer[j][i][1] = color[1]
                        buffer[j][i][2] = color[2]

def draw_line_on_buffer(buffer, image_w, image_h, points_2d, color, size=4):
  x0 = int(points_2d[0][0])
  y0 = int(points_2d[0][1])
  x1 = int(points_2d[1][0])
  y1 = int(points_2d[1][1])
  dx = abs(x1 - x0)
  if x0 < x1:
    sx = 1
  else:
    sx = -1
  dy = -abs(y1 - y0)
  if y0 < y1:
    sy = 1
  else:
    sy = -1
  err = dx + dy
  while True:
    draw_points_on_buffer(buffer, image_w, image_h, ((x0,y0),), color, size)
    if (x0 == x1 and y0 == y1):
      break
    e2 = 2 * err
    if (e2 >= dy):
      err += dy
      x0 += sx
    if (e2 <= dx):
      err += dx
      y0 += sy

def draw_skeleton(buffer, image_w, image_h, boneIndex, points2d, color, size=4):
    try:
        # draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_root"]], points2d[boneIndex["crl_hips__C"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hips__C"]], points2d[boneIndex["crl_spine__C"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hips__C"]], points2d[boneIndex["crl_thigh__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hips__C"]], points2d[boneIndex["crl_thigh__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_spine__C"]], points2d[boneIndex["crl_spine01__C"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_spine01__C"]], points2d[boneIndex["crl_shoulder__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_spine01__C"]], points2d[boneIndex["crl_neck__C"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_spine01__C"]], points2d[boneIndex["crl_shoulder__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_shoulder__L"]], points2d[boneIndex["crl_arm__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_arm__L"]], points2d[boneIndex["crl_foreArm__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_foreArm__L"]], points2d[boneIndex["crl_hand__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__L"]], points2d[boneIndex["crl_handThumb__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__L"]], points2d[boneIndex["crl_handIndex__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__L"]], points2d[boneIndex["crl_handMiddle__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__L"]], points2d[boneIndex["crl_handRing__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__L"]], points2d[boneIndex["crl_handPinky__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handThumb__L"]], points2d[boneIndex["crl_handThumb01__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handThumb01__L"]], points2d[boneIndex["crl_handThumb02__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handThumb02__L"]], points2d[boneIndex["crl_handThumbEnd__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handIndex__L"]], points2d[boneIndex["crl_handIndex01__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handIndex01__L"]], points2d[boneIndex["crl_handIndex02__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handIndex02__L"]], points2d[boneIndex["crl_handIndexEnd__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handMiddle__L"]], points2d[boneIndex["crl_handMiddle01__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handMiddle01__L"]], points2d[boneIndex["crl_handMiddle02__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handMiddle02__L"]], points2d[boneIndex["crl_handMiddleEnd__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handRing__L"]], points2d[boneIndex["crl_handRing01__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handRing01__L"]], points2d[boneIndex["crl_handRing02__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handRing02__L"]], points2d[boneIndex["crl_handRingEnd__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handPinky__L"]], points2d[boneIndex["crl_handPinky01__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handPinky01__L"]], points2d[boneIndex["crl_handPinky02__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handPinky02__L"]], points2d[boneIndex["crl_handPinkyEnd__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_neck__C"]], points2d[boneIndex["crl_Head__C"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_Head__C"]], points2d[boneIndex["crl_eye__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_Head__C"]], points2d[boneIndex["crl_eye__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_shoulder__R"]], points2d[boneIndex["crl_arm__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_arm__R"]], points2d[boneIndex["crl_foreArm__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_foreArm__R"]], points2d[boneIndex["crl_hand__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__R"]], points2d[boneIndex["crl_handThumb__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__R"]], points2d[boneIndex["crl_handIndex__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__R"]], points2d[boneIndex["crl_handMiddle__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__R"]], points2d[boneIndex["crl_handRing__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_hand__R"]], points2d[boneIndex["crl_handPinky__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handThumb__R"]], points2d[boneIndex["crl_handThumb01__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handThumb01__R"]], points2d[boneIndex["crl_handThumb02__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handThumb02__R"]], points2d[boneIndex["crl_handThumbEnd__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handIndex__R"]], points2d[boneIndex["crl_handIndex01__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handIndex01__R"]], points2d[boneIndex["crl_handIndex02__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handIndex02__R"]], points2d[boneIndex["crl_handIndexEnd__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handMiddle__R"]], points2d[boneIndex["crl_handMiddle01__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handMiddle01__R"]], points2d[boneIndex["crl_handMiddle02__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handMiddle02__R"]], points2d[boneIndex["crl_handMiddleEnd__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handRing__R"]], points2d[boneIndex["crl_handRing01__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handRing01__R"]], points2d[boneIndex["crl_handRing02__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handRing02__R"]], points2d[boneIndex["crl_handRingEnd__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handPinky__R"]], points2d[boneIndex["crl_handPinky01__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handPinky01__R"]], points2d[boneIndex["crl_handPinky02__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_handPinky02__R"]], points2d[boneIndex["crl_handPinkyEnd__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_thigh__R"]], points2d[boneIndex["crl_leg__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_leg__R"]], points2d[boneIndex["crl_foot__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_foot__R"]], points2d[boneIndex["crl_toe__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_toe__R"]], points2d[boneIndex["crl_toeEnd__R"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_thigh__L"]], points2d[boneIndex["crl_leg__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_leg__L"]], points2d[boneIndex["crl_foot__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_foot__L"]], points2d[boneIndex["crl_toe__L"]]), color, size)
        draw_line_on_buffer(buffer, image_w, image_h, (points2d[boneIndex["crl_toe__L"]], points2d[boneIndex["crl_toeEnd__L"]]), color, size)
    except:
        pass

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def write_image(frame, id, buffer):
    # Save the image using Pillow module.
    img = Image.fromarray(buffer)
    img.save('_out/%s_%06d.png' % (id, frame))

def vector_norm(location):
    """
    Calculate the norm (magnitude) of a 3D vector.
    
    Args:
        vector (list or numpy.ndarray): A 3D vector represented as a list or numpy array.
        
    Returns:
        float: The norm of the vector.
    """
    x = np.array([location.x, location.y, location.z])
    return np.sqrt(x.dot(x))

def endpoint_location_from_euler(euler_angles, length):
    """
    Calculate the endpoint location of a vector given Euler angles and its length.

    Args:
        euler_angles (list or numpy.ndarray): Euler angles (in radians) representing rotation about x, y, and z axes.
        length (float): Length of the vector.

    Returns:
        numpy.ndarray: Endpoint location (x, y, z) of the vector.
    """
    alpha, beta, gamma = euler_angles
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

    # Calculate the endpoint location using trigonometric functions
    endpoint_x = length * (cos_alpha * cos_beta)
    endpoint_y = length * (sin_alpha * cos_beta)
    endpoint_z = length * (-sin_beta)
    
    return np.array([endpoint_z, endpoint_x, endpoint_y])

from scipy.spatial.transform import Rotation as R

def sum_euler_angles(euler1, euler2,mode='zyx'):
    # Convert Euler angles to rotation matrices
    R1 = R.from_euler(mode, euler1, degrees=True).as_matrix()
    R2 = R.from_euler(mode, euler2, degrees=True).as_matrix()
    
    # Multiply the rotation matrices
    R_sum = np.dot(R1, R2)
    
    # Convert the resulting rotation matrix back to Euler angles
    sum_euler = R.from_matrix(R_sum).as_euler(mode, degrees=True)
    
    return sum_euler

def inverse_euler_angles(euler_angles):
    # Extract roll, pitch, and yaw from the input vector
    roll, pitch, yaw = euler_angles

    # Convert angles from degrees to radians if needed
    roll, pitch, yaw = np.radians([roll, pitch, yaw])

    # Compute the inverse Euler angles
    inverse_roll = -roll
    inverse_pitch = -pitch
    inverse_yaw = -yaw

    # Wrap the angles to the range [-pi, pi] or [-180, 180] if using degrees
    inverse_roll = np.arctan2(np.sin(inverse_roll), np.cos(inverse_roll))
    inverse_pitch = np.arctan2(np.sin(inverse_pitch), np.cos(inverse_pitch))
    inverse_yaw = np.arctan2(np.sin(inverse_yaw), np.cos(inverse_yaw))

    # Convert angles back to degrees if needed
    inverse_roll, inverse_pitch, inverse_yaw = np.degrees([inverse_roll, inverse_pitch, inverse_yaw])

    # Return the inverse Euler angles as a vector
    return np.array([inverse_roll, inverse_pitch, inverse_yaw])

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--fov',
        default=60,
        type=int,
        help='FOV for camera')
    argparser.add_argument(
      '--res',
      metavar='WIDTHxHEIGHT',
      # default='1920x1080',
      default='800x600',
      help='window resolution (default: 800x600)')
    args = argparser.parse_args()
    
    args.width, args.height = [int(x) for x in args.res.split('x')]

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    world = client.get_world()

    # spawn a camera 
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", str(args.fov))
    camera = world.spawn_actor(camera_bp, carla.Transform())
    
    # spawn a pedestrian
    world.set_pedestrians_seed(1235)
    ped_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.0028"))
    trans = carla.Transform()
    trans.location = world.get_random_location_from_navigation()
    ped = world.spawn_actor(ped_bp, trans)
    
    #walker_controller_bp = world.get_blueprint_library().find('controller.ai.walkerbonecontrol')
    #control = world.spawn_actor(walker_controller_bp, carla.Transform(), ped)
    #controller.start()
    #controller.go_to_location(world.get_random_location_from_navigation())
    #controller.set_max_speed(1.7)

    # keep tracking of actors to remove
    actor_list.append(camera)
    actor_list.append(ped)
    #actor_list.append(controller)

    # get some attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    try:
        pool = Pool(processes=5)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera, fps=30) as sync_mode:
            
            # set the projection matrix
            K = build_projection_matrix(image_w, image_h, fov)

            blending = 0
            turning = 0
            first = True
            second = 2
            while True:
                if should_quit():
                    return
                clock.tick()


                #First get all bones
                all_bones = ped.get_bones() 
                    

                #Now Filter and retrieve only the ones we are interested in
                filter_bones = ['crl_thigh__R','crl_leg__R','crl_foreArm__R','crl_foreArm__L','crl_leg__L','crl_arm__L','crl_arm__R','crl_spine__C','crl_thigh__L','crl_shoulder__L','crl_shoulder__R','crl_neck__C']


                filtered_bones = [bone for bone in all_bones.bone_transforms if bone.name in filter_bones]
                #[  -6.89563443, -22.34184681,  17.18824018],



                new_pose_dict = {
                                ##Right leg
                                'crl_thigh__R': {'rotation':sum_euler_angles([56.7500721243956, -16.544053832953992, 25.02675923303483],inverse_euler_angles([15,0,180]),'zxy'), 'location': [0, 0, 0],'parent': 'crl_root'}, #[20.54360794876757, -5.943302146652839, -9.427805053372714],[180,-180,-25]
                                'crl_leg__R': {'rotation': sum_euler_angles([-31.293895780348507,-14.517304603659372, 29.67655402376227],inverse_euler_angles([0,0,-176]),'zxy'), 'location': [0, 0, 0],'parent': 'crl_thigh__R'},
                                
                                ##Left leg
                                'crl_thigh__L': {'rotation':sum_euler_angles([74.5533496219488, -9.83629245274992, 39.232600391572824],inverse_euler_angles([-15,0,0]),'zxy'), 'location': [0, 0, 0],'parent': 'crl_root'},
                                'crl_leg__L': {'rotation': sum_euler_angles([0.024881520125725802, 0.005220097453850754, 0.018406008288751004],inverse_euler_angles([0,0,-176]),'zxy'), 'location': [0, 0, 0],'parent':'crl_thigh__L'},
                                
                                ##Right arm
                                #'crl_shoulder__R': {'rotation': sum_euler_angles([0,0,0],[0,-180,0],'xyz'), 'location': [0, 0, 0],'parent': 'crl_shoulder__L'},
                                #'crl_arm__R': {'rotation': sum_euler_angles([0,0,0],[0,-180,0],'xyz'), 'location': [0, 0, 0],'parent': 'crl_shoulder__L'},
                                #'crl_foreArm__R': {'rotation':sum_euler_angles([4.738812315757723, 50.04208808972586, 42.94656797471609],[0,0,-180],'xyz'), 'location': [0, 0, 0],'parent': 'crl_shoulder__R'},

                                ##Left arm
                                #'crl_shoulder__L': {'rotation': sum_euler_angles([-62.64876103012018, 5.223609785809075, 4.273746616340836],inverse_euler_angles([-11,3,0]),'xyz'), 'location': [0, 0, 0],'parent': 'crl_shoulder__L'},
                                'crl_arm__L': {'rotation': sum_euler_angles([-15.586735638399256, 2.153065908618747, 0.9271226181481296],inverse_euler_angles([11.35, -1.66, -180]),'xyz'), 'location': [0,0,0],'parent': 'crl_shoulder__L'},
                                'crl_foreArm__L': {'rotation':sum_euler_angles([78.73643752256984, 166.60799795539603, 173.4785571627607],inverse_euler_angles([-15,0,0]),'xyz'), 'location': [0, 0, 0],'parent': 'crl_shoulder__R'},

                                #'crl_spine__C': {'rotation':sum_euler_angles([-9.527387472475734, 2.1170614985602634, 7.553654895143941],[0,0,0],'xyz'), 'location': [0, 0, 0],'parent': 'crl_root'},
                                #'crl_hips__C': {'rotation':sum_euler_angles([-9.527387472475734, 2.1170614985602634, 7.553654895143941],[0,0,0],'xyz'), 'location': [0, 0, 0],'parent': 'crl_root'},

                                 }
                

                arml_l = np.random.randint(-45,0)
                arml_r = np.random.randint(-90,0)
                forearm_r = np.random.randint(-30,60)
                new_pose_dict = {
                                ##Right leg
                                 ##Right leg
                                'crl_thigh__R': {'rotation':[0,0,np.random.randint(-60,60)], 'location': [0, 0, 0],'parent': 'crl_root'}, #[20.54360794876757, -5.943302146652839, -9.427805053372714],[180,-180,-25]
                                'crl_leg__R': {'rotation':[0,0,np.random.randint(0,90)], 'location': [0, 0, 0],'parent': 'crl_root'},
                                
                                ##Left leg
                                'crl_thigh__L': {'rotation':[0,0,np.random.randint(-60,60)], 'location': [0, 0, 0],'parent': 'crl_root'},
                                'crl_leg__L': {'rotation':[0,0,np.random.randint(0,90)], 'location': [0, 0, 0],'parent':'crl_root'},

                                ##Right arm
                                #'crl_shoulder__R': {'rotation':[0,0,0], 'location': [0, 0, 0],'parent': 'crl_root'},
                                #'crl_arm__R': {'rotation':[-45,45,180], 'location': [0, 0, 0],'parent': 'crl_root'}, # de 0 a 90 el ptich
                                'crl_foreArm__R': {'rotation':[0,forearm_r,0], 'location': [0, 0, 0],'parent': 'crl_root'},

                                ##Left arm
                                #'crl_shoulder__L': {'rotation':[0,0,90], 'location': [0, 0, 0],'parent': 'crl_root'},
                                'crl_arm__L': {'rotation':[arml_l,0,arml_l], 'location': [0, 0, 0],'parent': 'crl_root'},#de 45 a 45 el pitch
                                'crl_foreArm__L': {'rotation':[0,np.random.randint(-60,30),0], 'location': [0, 0, 0],'parent': 'crl_root'},

                                ##Spine
                                'crl_spine__C': {'rotation':[np.random.randint(-75,75),0,0], 'location': [0, 0, 0],'parent': 'crl_root'},

                                ##Neck
                                'crl_neck__C': {'rotation':[0,np.random.randint(-30,30),0], 'location': [0, 0, 0],'parent': 'crl_root'},

                }   
                
                ## Spine C probablemente no va por tema de usar el 7 como el cero
                ## Faltaria hacer puntos intermedios para los hombros
                ## zZZzZzZZ z
                
                modified_bones_dict = {}
                
                root_bone = [bone for bone in all_bones.bone_transforms if bone.name == 'crl_root'][0]
                og_offset = np.array([root_bone.world.rotation.pitch,root_bone.world.rotation.yaw,root_bone.world.rotation.roll])
                if(first):
                    print( [bone for bone in all_bones.bone_transforms if bone.name == 'crl_thigh__L'][0])

                    print( [bone for bone in all_bones.bone_transforms if bone.name == 'crl_thigh__R'][0])

                    #Set the new pose
                    new_bones = []
                    for filtered_bone in filtered_bones:
                        if  filtered_bone.name in new_pose_dict:
                            #Find the father bone. Here we must check if the parent was already set by us, or not. If not, we use coordinates from the list, in theory, if we move from outside to inside, rotations should work....
                            #if new_pose_dict[filtered_bone.name]['parent'] is not None:

                            r_new = new_pose_dict[filtered_bone.name]['rotation']
                            rotation_vector = sum_euler_angles(r_new, [filtered_bone.relative.rotation.pitch,filtered_bone.relative.rotation.yaw,filtered_bone.relative.rotation.roll],'zxy')

                            print("New bone rotation",rotation_vector)
                            limb_length = vector_norm(filtered_bone.relative.location)
                            print(filtered_bone.name)
                            print("Limb length:",limb_length)

                            # if(filtered_bone.name == 'crl_thigh__L'):
                            #     k = [-31.794040181041705, -83.002627380923, 47.95112933471604]
                                
                            #     pose_vector = endpoint_location_from_euler([math.radians(k[0]), math.radians(k[1]), math.radians(k[2])], limb_length)
                            #     new_loc =  [pose_vector[0], pose_vector[1],pose_vector[2]]

                            # elif(filtered_bone.name == 'crl_thigh__R'):
                            #     k = [22.606679844660363, 48.947052672823844, 37.80280524199994]
                            #     pose_vector = endpoint_location_from_euler([math.radians(k[0]), math.radians(k[1]), math.radians(k[2])], limb_length)
                            #     new_loc =  [pose_vector[0], pose_vector[1],pose_vector[2]]
                            # elif(filtered_bone.name == 'crl_spine__C'):
                            #     k = sum_euler_angles([-0.0, 9.449465258252909, -67.46234671804034],[0,0,0],'xyz')
                            #     pose_vector = endpoint_location_from_euler([math.radians(k[0]), math.radians(k[1]), math.radians(k[2])], limb_length)
                            #     new_loc =  [pose_vector[0], pose_vector[1],pose_vector[2]]
                            # else:
                            pose_vector = [0,0,0]
                            new_loc = [filtered_bone.relative.location.x + pose_vector[0], filtered_bone.relative.location.y + pose_vector[1], filtered_bone.relative.location.z + pose_vector[2]]

                            print("New pose location",pose_vector)
                            print("Original pose location",filtered_bone.relative.location)
                            print("Relative vector",new_loc)
                            new_pose_dict[filtered_bone.name] = {'rotation': carla.Rotation(*rotation_vector), 'location': carla.Location(*new_loc)}
                            bone_t = (filtered_bone.name, carla.Transform(rotation=new_pose_dict[filtered_bone.name]['rotation'], location=new_pose_dict[filtered_bone.name]['location']))
                            new_bones.append(bone_t)
                    
                    
                    #Set bones
                    ft = [-23.75842191, -38.39732904,-28.83638448]
                    fd = [-0.09499745, -0.42752922,-0.05589557]
                    
                    st = [-1.45234558,  0.77358378, 2.63925492]
                    sd = [ 0.0153916 ,  -0.4361881, -0.12572408 ]
                    tt = [ 43.09562088, -26.70386462, -18.69296002]
                    td = [-0.01794255, +0.40565217,+0.17686112]
                    fot = [-6.76935755, -7.0954407 ,  7.52290112]

                    first_tuple = ('crl_leg__R', carla.Transform(rotation=carla.Rotation(*st), location=carla.Location(*sd)))

                    third_tuple = ('crl_leg__R', carla.Transform(rotation=carla.Rotation(*tt), location=carla.Location(*td)))


                    bones_setlist = carla.WalkerBoneControlIn(new_bones)
                
                    ped.set_bones(bones_setlist)
                    first = False

                #print(ped.get_bones())
                # make some transition from custom pose to animation
                ped.blend_pose(1)
                all_bones = ped.get_bones()
                if second > 0:
                    second -= 1
                    print([bone for bone in all_bones.bone_transforms if bone.name == 'crl_root'][0])
                    print([bone for bone in all_bones.bone_transforms if bone.name == 'crl_thigh__R'][0])
                    print([bone for bone in all_bones.bone_transforms if bone.name == 'crl_leg__R'][0])


                # move the pedestrian
                blending += 0.015
                turning += 0.009

                # move camera around
                trans = ped.get_transform()
                x = math.cos(turning) * -3
                y = math.sin(turning) * 3
                trans.location.x += x
                trans.location.y += y
                trans.location.z = 2
                trans.rotation.pitch = -16
                trans.rotation.roll = 0
                trans.rotation.yaw = -360 * (turning/(math.pi*2))
                camera.set_transform(trans)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=5.0)

                # Draw the display.
                buffer = get_image_as_array(image_rgb)

                # get the pedestrian bones
                bones = ped.get_bones()
                
                # prepare the bones (get name and world position)
                boneIndex = {}  
                points = []
                for i, bone in enumerate(bones.bone_transforms):
                    boneIndex[bone.name] = i
                    points.append(bone.world.location)
                
                # project the 3d points to 2d screen
                points2d = get_screen_points(camera, K, image_w, image_h, points)

                # draw the skeleton lines
                draw_skeleton(buffer, image_w, image_h, boneIndex, points2d, (0, 255, 0), 2)

                # draw the bone points
                draw_points_on_buffer(buffer, image_w, image_h, points2d[1:], (255, 0, 0), 4)

                draw_image(display, buffer)
                # pool.apply_async(write_image, (snapshot.frame, "ped", buffer))

                # display.blit(font.render('%d bones' % len(points), True, (255, 255, 255)), (8, 10))

                pygame.display.flip()

    finally:
        # time.sleep(5)
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        pool.close()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
