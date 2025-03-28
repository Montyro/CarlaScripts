#!/usr/bin/env python

# ==============================================================================
#
# This script is used to generate a dataset using carla, for visual odometry
# 
#
# ==============================================================================


# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import queue
import pygame
import numpy as np
import math
import argparse
import yaml

########################### Change to the location of PythonAPI in your computer ####################
try:
    sys.path.append(glob.glob('D:/carla_project/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

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


def save_scan(scan,environment_ready,counter):
    max_scans = 4500
    if counter >= max_scans:
        exit()

    if environment_ready == True:
        scan.save_to_disk('_out/{}.ply' % scan.frame) # Save the scan
        counter = counter+1
    #Add to poses list


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []    

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

# Function to change rotations in CARLA from left-handed to right-handed reference frame
def rotation_carla(rotation):
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    return np.array([[cy*cp, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr],[-sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[sp, cp*sr, cp*cr]])

# Function to change translations in CARLA from left-handed to right-handed reference frame
def translation_carla(location):
    if isinstance(location, np.ndarray):
        return location*(np.array([[1],[1],[1]]))
    else:
        return np.array([location.x, location.y, location.z])

def main(config):
    actor_list = []
    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.
    counter = 0
    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)


        pygame.init() # To be able to exit
        
        
        clock = pygame.time.Clock()
        # Once we have a client we can retrieve the world that is currently
        # running.
        #print(client.get_available_maps())

        world = client.get_world()
        
        print("Current map: {}".format(world.get_map().name.split('/')[-1]))
        if(world.get_map().name.split('/')[-1] != config['map']):
            client.set_timeout(120.0)
            print("Loading a new map: {}".format(config['map']))
            world = client.load_world("/Game/Carla/Maps/"+config['map'])
            client.set_timeout(10.0)



        
        #############################################################################
        # Set up synchronous mode
        #############################################################################

        settings = world.get_settings()
        fps = 10
        settings.fixed_delta_seconds = (1.0 / fps) if fps > 0.0 else 0.0
        settings.synchronous_mode = True
        ready = False
        synchronous_master = True
        world.apply_settings(settings)

        #client.reload_world(False)

        respawn = False
        hybrid = False
        seed = config['seed']
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        tm_port = traffic_manager.get_port()
        if seed is not None:
            traffic_manager.set_random_device_seed(seed)

        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        
        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle.mercedes.coupe'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        camera_spawn = np.random.choice(np.arange(len(world.get_map().get_spawn_points())))
        start_pose = world.get_map().get_spawn_points()[camera_spawn]
        waypoint = world.get_map().get_waypoint(start_pose.location)


        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, start_pose)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True,tm_port)
        

        #######################################################################################
        # Spawn Vehicles and People
        #######################################################################################
        
        batch = []

        filterv = 'vehicle'
        generationv = 'All'
        blueprints = get_actor_blueprints(world, filterv, generationv)
        safe = False

        if safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            #blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            #blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            #blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        car_blueprints = blueprints
        motorbike_blueprints = blueprints
        bike_blueprints = blueprints
        truck_blueprints = blueprints
        #filter cars (remove motorbikes,bikes and trucks)
        car_blueprints = [x for x in car_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('microlino')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('cybertruck')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('sprinter')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('firetruck')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('ambulance')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('carlacola')]
        car_blueprints = [x for x in car_blueprints if not x.id.endswith('truck')]

        motorbike_blueprints = [x for x in motorbike_blueprints if x.id.endswith('low_rider') | x.id.endswith('ninja') | x.id.endswith('zx125') | x.id.endswith('yzf')]
        bike_blueprints = [x for x in bike_blueprints if x.id.endswith('omafiets') | x.id.endswith('century') | x.id.endswith('crossbike')]
        truck_blueprints = [x for x in truck_blueprints if x.id.endswith('firetruck') | x.id.endswith('carlacola') | x.id.endswith('ambulance') | x.id.endswith('truck')]


        #blueprints = [blueprint_library.filter('kitti')]
        spawn_points = world.get_map().get_spawn_points()[:camera_spawn] + world.get_map().get_spawn_points()[camera_spawn+1:]

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        number_of_vehicles = config['vehicle_amounts'][0]['cars'] +config['vehicle_amounts'][2]['bikes'] + config['vehicle_amounts'][3]['trucks']+ config['vehicle_amounts'][1]['motorbikes']
        hero = False
        sensor_placed = False
 
        vehicle_amounts = {'car':(config['vehicle_amounts'][0]['cars'],car_blueprints),'bike':(config['vehicle_amounts'][2]['bikes'],bike_blueprints),'truck':(config['vehicle_amounts'][3]['trucks'],truck_blueprints),'motorbike':(config['vehicle_amounts'][1]['motorbikes'],motorbike_blueprints)} #must add to spawn points
        

        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break

            veh_type = random.choice(list(vehicle_amounts.keys()))
            blueprint = random.choice(vehicle_amounts[veh_type][1])

            vehicle_amounts[veh_type] = (vehicle_amounts[veh_type][0]-1,vehicle_amounts[veh_type][1])

            if vehicle_amounts[veh_type][0] <= 0:
                vehicle_amounts.pop(veh_type)

            print(blueprint.id)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            if sensor_placed:# spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            else:
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                print("Spawn Error")
            else:
                actor_list.append(response.actor_id)

        used_spawn = []

        for i in range(0, 15):
            #transform = world.get_map().get_spawn_points()[i+1+20]

            bp = random.choice(blueprint_library.filter('walker'))
            controller = blueprint_library.find('controller.ai.walker')
            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            trans = carla.Transform()
            got_location = world.get_random_location_from_navigation()
            while got_location in used_spawn:
                got_location = world.get_random_location_from_navigation()

            used_spawn.append(got_location)

            
            trans.location = got_location
            trans.location.z+=1
            npc = world.try_spawn_actor(bp, trans)
            world.tick()

            actor_controllers = []
            
            if npc is not None:
                controller_sp = world.spawn_actor(controller,carla.Transform(),npc)
                world.tick()
                controller_sp.start()
                controller_sp.go_to_location(world.get_random_location_from_navigation())
                actor_controllers.append(controller_sp)
                actor_controllers.append(npc)
                print('created %s' % npc.type_id)

        traffic_manager.global_percentage_speed_difference(10.0)


        #####################################
        # Traffic lights
        #####################################

        list_actor = world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is 
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                actor_.set_state(carla.TrafficLightState.Green) 
                actor_.set_green_time(100.0)

                # actor_.set_green_time(5000.0)
                # actor_.set_yellow_time(1000.0)

        ####################################################################################
        # Sensor initialization
        ####################################################################################
        sensor_list = []
        #Common parameters
        image_size_x = 1920
        image_size_y = 1080
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.63))#, carla.Rotation(pitch=-15))

       #### RGB Cameras ####
        # RGB Camera 1
        # Find blueprint
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        #Configure camera parameters
        camera_bp.set_attribute('fov',str(40)) #In cm
        camera_bp.set_attribute('image_size_x',str(image_size_x))
        camera_bp.set_attribute('image_size_y',str(image_size_y))

        #RGB Camera 2
        # Find blueprint
        camera_bp_2 = blueprint_library.find('sensor.camera.rgb')
        #Configure camera parameters
        camera_bp_2.set_attribute('fov',str(60)) #In cm
        camera_bp_2.set_attribute('image_size_x',str(image_size_x))
        camera_bp_2.set_attribute('image_size_y',str(image_size_y))
         #RGB Camera 3
        # Find blueprint
        camera_bp_3 = blueprint_library.find('sensor.camera.rgb')
        #Configure camera parameters
        camera_bp_3.set_attribute('fov',str(80)) #In cm
        camera_bp_3.set_attribute('image_size_x',str(image_size_x))
        camera_bp_3.set_attribute('image_size_y',str(image_size_y))
         #RGB Camera 4
        # Find blueprint
        camera_bp_4 = blueprint_library.find('sensor.camera.rgb')
        #Configure camera parameters
        camera_bp_4.set_attribute('fov',str(100)) #In cm
        camera_bp_4.set_attribute('image_size_x',str(image_size_x))
        camera_bp_4.set_attribute('image_size_y',str(image_size_y))
         #RGB Camera 5
        # Find blueprint
        camera_bp_5 = blueprint_library.find('sensor.camera.rgb')
        #Configure camera parameters
        camera_bp_5.set_attribute('fov',str(120)) #In cm
        camera_bp_5.set_attribute('image_size_x',str(image_size_x))
        camera_bp_5.set_attribute('image_size_y',str(image_size_y))

        #Spawn the camera sensor
        camera_rgb_1 = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle)
        camera_rgb_2 = world.spawn_actor(
            camera_bp_2,
            camera_transform,
            attach_to=vehicle)
        camera_rgb_3 = world.spawn_actor(
            camera_bp_3,
            camera_transform,
            attach_to=vehicle)
        camera_rgb_4 = world.spawn_actor(
            camera_bp_4,
            camera_transform,
            attach_to=vehicle)
        camera_rgb_5 = world.spawn_actor(
            camera_bp_5,
            camera_transform,
            attach_to=vehicle)

        sensor_list.append(camera_rgb_1)
        sensor_list.append(camera_rgb_2)
        sensor_list.append(camera_rgb_3)
        sensor_list.append(camera_rgb_4)
        sensor_list.append(camera_rgb_5)



        #### Depth Cameras ####
        # Depth Camera 1
        # Find blueprint
        depth_camera_bp = blueprint_library.find('sensor.camera.depth')
        #Configure camera parameters
        depth_camera_bp.set_attribute('fov',str(40)) #In cm
        depth_camera_bp.set_attribute('image_size_x',str(image_size_x))
        depth_camera_bp.set_attribute('image_size_y',str(image_size_y))

        #Depth Camera 2
        # Find blueprint
        depth_camera_bp_2 = blueprint_library.find('sensor.camera.depth')
        #Configure camera parameters
        depth_camera_bp_2.set_attribute('fov',str(60)) #In cm
        depth_camera_bp_2.set_attribute('image_size_x',str(image_size_x))
        depth_camera_bp_2.set_attribute('image_size_y',str(image_size_y))
        
        #Depth Camera 3
        # Find blueprint
        depth_camera_bp_3 = blueprint_library.find('sensor.camera.depth')
        #Configure camera parameters
        depth_camera_bp_3.set_attribute('fov',str(80)) #In cm
        depth_camera_bp_3.set_attribute('image_size_x',str(image_size_x))
        depth_camera_bp_3.set_attribute('image_size_y',str(image_size_y))

        #Depth Camera 4
        # Find blueprint
        depth_camera_bp_4 = blueprint_library.find('sensor.camera.depth')
        #Configure camera parameters
        depth_camera_bp_4.set_attribute('fov',str(100)) #In cm
        depth_camera_bp_4.set_attribute('image_size_x',str(image_size_x))
        depth_camera_bp_4.set_attribute('image_size_y',str(image_size_y))

        #Depth Camera 5
        # Find blueprint
        depth_camera_bp_5 = blueprint_library.find('sensor.camera.depth')
        #Configure camera parameters
        depth_camera_bp_5.set_attribute('fov',str(120)) #In cm
        depth_camera_bp_5.set_attribute('image_size_x',str(image_size_x))
        depth_camera_bp_5.set_attribute('image_size_y',str(image_size_y))

        #Spawn the camera sensor
        depth_camera_1 = world.spawn_actor(
            depth_camera_bp,
            camera_transform,
            attach_to=vehicle)
        depth_camera_2 = world.spawn_actor(
            depth_camera_bp_2,
            camera_transform,
            attach_to=vehicle)
        depth_camera_3 = world.spawn_actor(
            depth_camera_bp_3,
            camera_transform,
            attach_to=vehicle)
        depth_camera_4 = world.spawn_actor(
            depth_camera_bp_4,
            camera_transform,
            attach_to=vehicle)
        depth_camera_5 = world.spawn_actor(
            depth_camera_bp_5,
            camera_transform,
            attach_to=vehicle)
        
        sensor_list.append(depth_camera_1)
        sensor_list.append(depth_camera_2)
        sensor_list.append(depth_camera_3)
        sensor_list.append(depth_camera_4)
        sensor_list.append(depth_camera_5)


        dataset_path = 'D:/odometry/seq05/'

        first_frame = True
        with open(dataset_path+"poses.txt", 'w') as posfile:
             posfile.write("## {} {} {} {} {} {}".format("roll","pitch","yaw","x","y","z"))



        ##############################################################################################
        # Create a synchronous mode context.
        ##############################################################################################
        with CarlaSyncMode(world, camera_rgb_1,camera_rgb_2,camera_rgb_3,camera_rgb_4,camera_rgb_5,depth_camera_1,depth_camera_2,depth_camera_3,depth_camera_4,depth_camera_5, fps=10) as sync_mode:
            counter = 0
            while True:
                if should_quit():
                    return
                clock.tick()

                
                # Advance the simulation and wait for the data.
                snapshot, rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,depth_1,depth_2,depth_3,depth_4,depth_5 = sync_mode.tick(timeout=10.0) #Ajusta timeout si el pc es muy lento
                if first_frame:
                    #initial_camera_position = vehicle.get_location() + camera_transform.location
                    initial_camera_rotation = vehicle.get_transform().rotation
                    initial_camera_position = actor_list[0].get_location() + carla.Location(z=1.63)
                    print("Initial camera position (absolute): {}".format(initial_camera_position))
                    first_frame = False

                # Choose the next waypoint and update the car location.
                #waypoint = random.choice(waypoint.next(1.5))
                #vehicle.set_transform(waypoint.transform)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Lets save position and rotation (roll pitch yaw) as global values and will process them later...
                #rotation_v = np.matmul(rotation_carla(initial_camera_rotation).T,rotation_carla(lidar_scan.transform.rotation))
                #translation_v = np.array(translation_carla(vehicle.get_location() + carla.Location(z=1.63)) - translation_carla(initial_camera_position))
                #print(translation_v)

                # Save the scans
                counter+=1

                cc = carla.ColorConverter.Depth
                cc2 = carla.ColorConverter.LogarithmicDepth

                rgb_1.save_to_disk(dataset_path+('cam1/rgb/{}.png').format(rgb_1.frame)) # Save the scan
                rgb_2.save_to_disk(dataset_path+('cam2/rgb/{}.png').format(rgb_2.frame)) # Save the scan
                rgb_3.save_to_disk(dataset_path+('cam3/rgb/{}.png').format(rgb_3.frame)) # Save the scan
                rgb_4.save_to_disk(dataset_path+('cam4/rgb/{}.png').format(rgb_4.frame)) # Save the scan
                rgb_5.save_to_disk(dataset_path+('cam5/rgb/{}.png').format(rgb_5.frame)) # Save the scan

                depth_1.save_to_disk(dataset_path+('cam1/depth/{}.png').format(depth_1.frame)) # Save the scan
                depth_2.save_to_disk(dataset_path+('cam2/depth/{}.png').format(depth_2.frame)) # Save the scan
                depth_3.save_to_disk(dataset_path+('cam3/depth/{}.png').format(depth_3.frame)) # Save the scan
                depth_4.save_to_disk(dataset_path+('cam4/depth/{}.png').format(depth_4.frame)) # Save the scan
                depth_5.save_to_disk(dataset_path+('cam5/depth/{}.png').format(depth_5.frame)) # Save the scan
                #image_rgb_4.save_to_disk('_out_rgb4/{}.png' % image_rgb_4.frame) # Save the scan
                #image_rgb_5.save_to_disk('_out_rgb5/{}.png' % image_rgb_5.frame) # Save the scan

                
                #Save poses
                with open(dataset_path+"poses.txt", 'a') as posfile:
                    posfile.write("{} {} {} {} {} {}".format(rgb_1.transform.rotation.roll,rgb_1.transform.rotation.pitch,rgb_1.transform.rotation.yaw,rgb_1.transform.location.x,rgb_1.transform.location.y,rgb_1.transform.location.z))
                    #posfile.write(" ".join(map(str,[r for r in rotation_v[0]]))+" "+str(translation_v[1])+" ")
                    #posfile.write(" ".join(map(str,[r for r in rotation_v[1]]))+" "+str(translation_v[0])+" ")
                    #posfile.write(" ".join(map(str,[r for r in rotation_v[2]]))+" "+str(translation_v[2])+" ")
                    posfile.write("\n")

                if counter >= config['sampling_steps']:
                    return
    finally:
        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(actor_controllers), 2):
            actor_controllers[i].stop()

        for i in sensor_list:
            i.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_controllers[::2]])

        time.sleep(0.5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CARLA Visual Odometry Dataset Generator')
    parser.add_argument('--config_file','-c', help='Config file',required=True)
    parser.add_argument('--output_folder','-o', help='Output folder',required=False)
    
    args = parser.parse_args()

    with(open(args.config_file)) as f:
        config = yaml.safe_load(f)

    main(config)
