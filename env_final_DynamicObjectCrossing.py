 #!/usr/bin/env python

import glob
import os
import sys
import re
import weakref
import carla
import pygame
import random
import time
import subprocess
from carla import ColorConverter as cc
# import psutil
import math
# import matplotlib.pyplot as plt
import numpy as np
import gym
import atexit
# from datetime import datetime
from gym.spaces import Box, Discrete, Tuple
from scipy.stats import multivariate_normal
import os
import signal
import datetime
from datetime import timedelta
import psutil
from planet import ENV_CONFIG
from planet import PID_FILE_NAME
# Default environment configuration

#9 kinds of scenario

from srunner.scenarios.follow_leading_vehicle import *
from srunner.scenarios.opposite_vehicle_taking_priority import *
from srunner.scenarios.object_crash_vehicle import *
from srunner.scenarios.no_signal_junction_crossing import *
from srunner.scenarios.object_crash_intersection import *
from srunner.scenarios.control_loss import *
from srunner.scenarios.config_parser import *
from srunner.scenariomanager.scenario_manager import ScenarioManager

from scenario_runner import *




import threading
""" default is rgb 
    stack for gray depth segmentation stack together
    encode for encode measurement in forth channel """

live_carla_processes = set()
# pid = os.getpid()
# print(pid, "<<<<<<<<<<<<<<<<<<\n"*200)
def cleanup():
    def stop(pid):
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    live_carla_processes = np.loadtxt(PID_FILE_NAME, dtype=int, ndmin=1)
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        try:
            # os.killpg(pgid, signal.SIGKILL)
            # os.kill(pgid, 9)
            stop(pgid)
        except:
            pass

atexit.register(cleanup)

COUNT = 0

class CarlaEnv(gym.Env):
    def __init__(self, config=ENV_CONFIG):
        self.config = config
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'*200)
        self.command = {
            "stop": 1,
            "lane_keep": 2,
            "turn_right": 3,
            "turn_left": 4,
        }

        # change action space
        self.action_space = Box(-1.0, 1.0, shape=(ENV_CONFIG["action_dim"], ), dtype=np.float32)

        if ENV_CONFIG["image_mode"] == "encode":
            framestack = 7
        elif ENV_CONFIG["image_mode"] == "stack":
            framestack = 3
        else:
            framestack = 3

        image_space = Box(
            0,
            255,
            shape=(config["y_res"], config["x_res"], framestack),
            dtype=np.float32)
        self.observation_space = image_space
        # environment config
        self._spec = lambda: None
        self._spec.id = "Carla_v0"
        # experiment config
        self.num_steps = 0
        self.total_reward = 0
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.feature_map = None
        # actors
        self.actor_list = []          # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.camera_rgb1 = None
        self.camera_rgb2 = None
        self.invasion_sensor = None
        # states and data
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_rgb1 = []          # save a list of rgb image
        self._image_rgb2 = []          # save a list of rgb image
        self._history_waypoint = []
        self._obs_collect = []
        self._global_step = 0
        # # self._d_collect = []
        # # initialize our world
        # self._carla_server = ServerManagerBinary()
        # self.server_port = random.randint(1000, 60000)
        # self.world = None
        #
        # # start a new carla service
        # self._carla_server.reset(self.config["host"], self.server_port)
        # self._carla_server.wait_until_ready()
        self.server_process = None
        self.server_port = None
        self.world = None
        # self.init_server()
        self._error_rest_test = 0

    def __del__(self):
        cleanup()

    def init_server(self):
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = 2000
        self.server_process = subprocess.Popen(
            [
                "/home/sdc/Desktop/carla94/CarlaUE4.sh", "/Game/Carla/Maps/Town03", "-benchmark", '-fps=20'
                "-ResX=1024","-ResY=768", "-host = 127.0.0.1", "-carla-port=2000",
            ],
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"))
        live_carla_processes.add(self.server_process.pid)
        # print(live_carla_processes)
        # live_carla_processes.add(os.getpgid(self.server_process.pid))
        try:
            pre_pid = np.loadtxt(PID_FILE_NAME, ndmin=1)
            pre_pid = pre_pid.astype(int)
            if len(pre_pid)>5:
                pre_pid = np.delete(pre_pid, range(0, len(pre_pid - 5)))
        except:
            pre_pid = []
        pid = np.array([x for x in live_carla_processes])
        np.savetxt(PID_FILE_NAME, np.concatenate([pre_pid, pid]), fmt='%d')
        # with open('/tmp/_carla_pid.txt', 'w') as f:
        #     f.write(str(self.server_process.pid))
            # f.write(str(os.getpgid(self.server_process.pid)))   # write carla server pid into file
        time.sleep(20)  # wait for world get ready

    # @set_timeout(10)
    def _restart(self):
        """restart world and add sensors"""
        # self.init_server()
        connect_fail_times = 0
        self.world = None
        while self.world is None:
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(2.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
            except Exception as e:
                connect_fail_times += 1
                print("Error connecting: {}, attempt {}".format(e, connect_fail_times))
                time.sleep(2)
            if connect_fail_times > 5:
                break

        world = self.world
        self._global_step = 0
        # actors
        self.actor_list = [] # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.invasion_sensor = None
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_rgb1 = []         # save a list of rgb image
        self._image_rgb2 = []
        self._history_waypoint = []
        self.other_actors_list = []   #save other actors
        
        for a in self.world.get_actors().filter('vehicle.*'):
            # print(a)
            try:
                a.destroy()
                time.sleep(2)
            except:
                pass
        # for b in self.world.get_actors().filter('sensor.*'):
        #     try:
        #         b.destroy()
        #         time.sleep(2)
        #     except:
        #         pass

        try:
            #setup ego_vehicle & other_actor
            self.scenario_name = 'DynamicObjectCrossing'     #the name of scenario
            bp_library = self.world.get_blueprint_library()
            # setup vehicle
            scenario_config_file = find_scenario_config(self.scenario_name)
            self.scenario_configurations = parse_scenario_configuration(scenario_config_file, self.scenario_name)

            for config in self.scenario_configurations:
                print(config)
                # Prepare scenario
                print("Preparing scenario: " + config.name)


                scenario_class = ScenarioRunner.get_scenario_class_or_fail(config.type)

                try:
                    # init ego_vehicle

                    # spawn_point of ego_vehicle
                    spawn_point = config.ego_vehicle.transform
                    # model of ego_vehicle
                    bp_vehicle = config.ego_vehicle.model
                    blueprint = random.choice(bp_library.filter(bp_vehicle))
                    blueprint.set_attribute('role_name', 'hero')

                    # set ego_vehicle
                    self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                    self.actor_list.append(self.vehicle)

                    # init other_actor
                    for actor in config.other_actors:

                        # spawn_point of other actor
                        actor_point = actor.transform
                        # model of other actor
                        actor_bp_vehicle = actor.model
                        actor_blueprint = random.choice(bp_library.filter(actor_bp_vehicle))
                        actor_blueprint.set_attribute('role_name', 'scenario')
                        self.other_actor = self.world.try_spawn_actor(actor_blueprint, actor_point)

                        #set other actor
                        self.other_actors_list.append(self.other_actor)
                        self.actor_list.append(self.other_actors_list)

                        #input parameter of scenario
                    self.scenario = scenario_class(self.world,
                                                    self.vehicle,
                                                    self.other_actors_list,
                                                    config.town,
                                                    False,
                                                    False)
                except Exception as exception:
                    print("The scenario cannot be loaded")
                    traceback.print_exc()
                    print(exception)
                    self.cleanup()

            #setup sensor
            camera_transform = carla.Transform(carla.Location(x=1, y=0, z=2))
            camera_rgb1 = bp_library.find('sensor.camera.rgb')
            camera_rgb1.set_attribute('fov', '120')
            camera_rgb1.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
            camera_rgb1.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
            # here is the bug, carla will throw bad_weak_ptr() and block the process
            self.camera_rgb1 = world.spawn_actor(camera_rgb1, camera_transform, attach_to=self.vehicle) # 32 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.actor_list.append(self.camera_rgb1)
            bp = bp_library.find('sensor.other.collision')
            self.collision_sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(self.collision_sensor)
            bp = bp_library.find('sensor.other.lane_detector')
            self.invasion_sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(self.invasion_sensor)       # 39 steps for first time 42 steps for reset
        except Exception as e:
            print("spawn fail, sad news", e)



    def reset(self):
        error = None
        for _ in range(100):
            try:
                if len(live_carla_processes) == 0:
                    self.init_server()
                self._restart()  # bugggggggggg!!!!!!!!!!!!!!!!!!!!!!!!
                obs = self._reset()
                return obs
            except Exception as e:
                with open("/home/gu/error_log %s.txt" % str(datetime.datetime.now()), "w") as f:
                    f.write('============Error====================, %s' % str(e))
                print("<<<<<<<<<<Error during reset in env>>>>>>>>>>")
                cleanup()
                self.init_server()
                error = e
        raise error

   #  @set_timeout(10)
    def _reset(self):
        # self._error_rest_test += 1
        # if self._error_rest_test < 3:
        #     print(1/0)
        # else:
        #     print("+++++++++++++++++++++++++++++++++++++++++++++++")
        weak_self = weakref.ref(self)
        # set invasion sensor
        self.invasion_sensor.listen(lambda event: self._parse_invasion(weak_self, event))
        # set collision sensor
        self.collision_sensor.listen(lambda event: self._parse_collision(weak_self, event))
        # set rgb camera sensor
        self.camera_rgb1.listen(lambda image: self._parse_image1(weak_self, image, cc.Raw, 'rgb'))
        while len(self._image_rgb1) < 4:
            print("resetting rgb")
            time.sleep(0.001)
        if ENV_CONFIG["image_mode"] == "encode":   # stack gray depth segmentation
            obs = np.concatenate([self._image_rgb1[-1], self._image_rgb1[-2],
                                  np.zeros([ENV_CONFIG['x_res'], ENV_CONFIG['y_res'], 1])], axis=2)
        else:
            obs = self._image_rgb1[-1]

        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        acceleration = self.vehicle.get_acceleration()
        if len(self._history_invasion) > 0:
            invasion = self._history_invasion[-1]
        else:
            invasion = []
        self.planner()
        distance = ((self._history_waypoint[-1].transform.location.x - self.vehicle.get_location().x)**2 + 
                   (self._history_waypoint[-1].transform.location.y - self.vehicle.get_location().y)**2)**0.5

        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                "acceleration": math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),
                "location_x": t.location.x,
                "location_y": t.location.y,
                "Throttle": c.throttle,
                "Steer": c.steer,
                "Brake": c.brake,
                "command": self.planner(),
                "distance": distance,
                "lane_invasion": invasion,
                "traffic_light": str(self.vehicle.get_traffic_light_state()),    # Red Yellow Green Off Unknown
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),       # True False
                "collision": len(self._history_collision)}

        self._history_info.append(info)
        self._obs_collect.append(obs[:, :, 0:3])
        if len(self._obs_collect) > 32:
            self._obs_collect.pop(0)
        mask = self._compute_mask()
        # define how many channel we want play with
        if ENV_CONFIG["attention_mode"] == "soft":
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] + mask
        else:
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] * mask
        self._obs_collect.append(np.clip(obs, 0, 255))  # clip in case we want render
        if len(self._obs_collect) > 32:
            self._obs_collect.pop(0)
        return self._obs_collect[-1]

    @staticmethod
    def _generate_point_list():
        """
        generate the Cartesian coordinates for every pixel in the picture, because attention point is represented in
        Cartesian coordinates(e.g. (-48, -48) (0, 0) (48, 48)) but the position of pixel is represented by index(e.g.
        [95, 0] [47, 47] [0 95])
        :return: Cartesian coordinates for pixels
        """
        r = int(ENV_CONFIG["x_res"]/2)
        point_list = []
        for i in range(r, -r, -1):
            for j in range(-r, r, 1):
                point_list.append((j, i))
        return point_list

    @staticmethod
    def _compute_distance_transform(d, action=np.zeros(ENV_CONFIG["action_dim"])):
        """compute the variance for attention mask when we adding noise
        if we specify attention mode to soft we will use this function """
        if ENV_CONFIG["action_dim"] == 5:
            # in care our poor agent see nothing we set threshold equal to 5
            # in other word if action[4] = 0 then action[4] will be set to 5
            # action[4] belong to range(-1, 1) we project it to [0, 70]
            r = 35*(1+action[4]) if 35*(1+action[4]) > 5 else 5
        else:
            r = 25
        if ENV_CONFIG["attention_mode"] == "soft":
            # d is the threshold of distance between attention point
            # if the distance is greater then d we add noise on image
            # the strength of noise is linear to distance
            d = 0 if d < r else 2 * d
        elif ENV_CONFIG["attention_mode"] == "hard":
            # it behave like mask(i.e. 0 for totally dark)
            d = 1 if d < r else (r/d)**2.5
        # d = -24 + 2*d
        return d

    def _compute_mask(self, action=np.zeros(ENV_CONFIG["action_dim"])):
        """"compute mask for attention"""
        if ENV_CONFIG["action_dim"] == 4 or ENV_CONFIG["action_dim"] == 5:
            mu_1 = int(ENV_CONFIG["x_res"] * action[2] * 0.5)
            mu_2 = int(ENV_CONFIG["y_res"] * action[3] * 0.5)
        elif ENV_CONFIG["action_dim"] == 2:
            mu_1 = 0
            mu_2 = 0
        d_list = []
        point_list = self._generate_point_list()
        for p in point_list:
            d = np.sqrt((mu_1 - p[0]) ** 2 + (mu_2 - p[1]) ** 2)
            if ENV_CONFIG["attention_mode"] == "soft":
                # self._d_collect.append(d)
                p_mask = float(self._compute_distance_transform(d, action) * np.random.randn(1))
            elif ENV_CONFIG["attention_mode"] == "hard":
                p_mask = float(self._compute_distance_transform(d, action))
            else:  # if we want use raw rgb
                p_mask = 1
            d_list.append(p_mask)
        mask = np.reshape(d_list, [ENV_CONFIG["x_res"], ENV_CONFIG["y_res"]])
        return mask[:, :, np.newaxis]

    @staticmethod
    def _parse_image1(weak_self, image, cc, use):
        """convert BGRA to RGB"""
        self = weak_self()
        if not self:
            return

        def convert(cc):
            image.convert(cc)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, -2:-5:-1]
            array = array.astype(np.float32)
            return array

        if use == 'rgb':
            array = convert(cc)
            self._image_rgb1.append(array)
            if len(self._image_rgb1) > 32:
                self._image_rgb1.pop(0)

    @staticmethod
    def _parse_image2(weak_self, image, cc, use):
        """convert BGRA to RGB"""
        self = weak_self()
        if not self:
            return

        def convert(cc):
            image.convert(cc)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, -2:-5:-1]
            array = array.astype(np.float32)
            return array

        if use == 'rgb':
            array = convert(cc)
            self._image_rgb2.append(array)
            if len(self._image_rgb2) > 32:
                self._image_rgb2.pop(0)

    @staticmethod
    def _parse_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history_collision.append((event.frame_number, intensity))
        if len(self._history_collision) > 32:
            self._history_collision.pop(0)

    @staticmethod
    def _parse_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        # print(str(event.crossed_lane_markings)) [carla.libcarla.LaneMarking.Solid]
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        # S for Solid B for Broken
        self._history_invasion.append(text[0][1])
        if len(self._history_invasion) > 32:
             self._history_invasion.pop(0)

    def step(self, action):
        try:
            obs = self._step(action)
            return obs
        except Exception as e:
            print("Error during step, terminating episode early")
            print(e)
        return self._obs_collect[-1], 0, True, self._history_info[-1]

    # @set_timeout(10)
    def _step(self, action):
        self._global_step += 1

        def compute_reward(info, prev_info):
            reward = 0.0
            reward += np.clip(info["speed"], 0, 15)/3
            reward += info['distance']
            if info["collision"] == 1:
                reward -= 70
            elif 2 <= info["collision"] < 5:
                reward -= info['speed'] * 2
            elif info["collision"] > 5:
                reward -= info['speed'] * 1

            print(self._global_step, "current speed", info["speed"], "collision", info['collision'])
            new_invasion = list(set(info["lane_invasion"]) - set(prev_info["lane_invasion"]))
            if 'S' in new_invasion:     # go across solid lane
                 reward -= info["speed"]
            elif 'B' in new_invasion:   # go across broken lane
                 reward -= 0.4 * info["speed"]
            return reward

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        distance_before_act = ((self._history_waypoint[-1].transform.location.x - self.vehicle.get_location().x)**2 + 
                   (self._history_waypoint[-1].transform.location.y - self.vehicle.get_location().y)**2)**0.5
      
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        # sleep a little waiting for the responding from simulator
        if ENV_CONFIG["attention_mode"] == "None": #  or ENV_CONFIG["attention_mode"] == "hard":
          time.sleep(0.04)

        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        acceleration = self.vehicle.get_acceleration()
        if len(self._history_invasion) > 0:
            invasion = self._history_invasion[-1]
        else:
            invasion = []
             
        command = self.planner()
       
        distance_after_act = ((self._history_waypoint[-2].transform.location.x - self.vehicle.get_location().x)**2 + 
                              (self._history_waypoint[-2].transform.location.y - self.vehicle.get_location().y)**2)**0.5
        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                "acceleration": math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),
                "location_x": t.location.x,
                "location_y": t.location.y,
                "Throttle": c.throttle,
                "Steer": c.steer,
                "Brake": c.brake,
                "command": command,
                "distance": distance_before_act - distance_after_act,  # distance to waypoint
                "lane_invasion": invasion,
                "traffic_light": str(self.vehicle.get_traffic_light_state()),    # Red Yellow Green Off Unknown
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),       # True False
                "collision": len(self._history_collision)}

        self._history_info.append(info)
        reward = compute_reward(self._history_info[-1], self._history_info[-2])
        # print(self._history_info[-1]["speed"], self._history_info[-1]["collision"])

        # early stop
        done = False
        if ENV_CONFIG["early_stop"]:
            if len(self._history_collision) > 0 and self._global_step > 60:
                # print("collisin length", len(self._history_collision))
                done = True
                # self.destroy_actor()
            # elif reward < -100:
            #     done = True

        if ENV_CONFIG["image_mode"] == "encode":   # stack gray depth segmentation
            obs = np.concatenate([self._image_rgb1[-1], self._image_rgb1[-2],
                                  self.encode_measurement(info)], axis=2)
        else:
            obs = self._image_rgb1[-1]

        mask = self._compute_mask(action)
        if ENV_CONFIG["attention_mode"] == "soft":
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] + mask
        else:
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] * mask

        self._obs_collect.append(np.clip(obs, 0, 255))  # clip in case we want render
        if len(self._obs_collect) > 32:
            self._obs_collect.pop(0)

        return self._obs_collect[-1], reward, done, self._history_info[-1]

    def render(self):
        display = pygame.display.set_mode(
            (ENV_CONFIG["x_res"], ENV_CONFIG["y_res"]),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        # surface = pygame.surfarray.make_surface(env._image_rgb1[-1].swapaxes(0, 1))
        surface = pygame.surfarray.make_surface(env._obs_collect[-1][:,:,0:3].swapaxes(0, 1))
        display.blit(surface, (0, 0))
        time.sleep(0.01)
        pygame.display.flip()

    def planner(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        waypoint = random.choice(waypoint.next(12.0))
        self._history_waypoint.append(waypoint)
        yaw = waypoint.transform.rotation.yaw
        if yaw > -90 or yaw < 60:
            command = "turn_right"
        elif yaw > 60 and yaw < 120:
            command = "lane_keep"
        elif yaw > 120 or yaw < -90:
            command = "turn_left"
        return self.command[command]

    @staticmethod
    def encode_measurement(py_measurements):
        """encode measurements into another channel"""
        feature_map = np.zeros([4, 4])
        feature_map[0, :] = (py_measurements["command"]) * 60.0
        feature_map[1, :] = (py_measurements["speed"]) * 4.0
        feature_map[2, :] = (py_measurements["command"]) * 60.0
        feature_map[3, :] = (py_measurements["Steer"]+1) * 120.0
        stack = int(ENV_CONFIG["x_res"]/4)
        feature_map = np.tile(feature_map, (stack, stack))
        feature_map = feature_map.astype(np.float32)
        return feature_map[:, :, np.newaxis]


    # load the scenario
    def load_scenario(self):

        self.manager = ScenarioManager(self.world, False)
        self.manager.load_scenario(self.scenario)

    # run the scenario
    def run_scenario(self):
        self.manager.run_scenario()

    # analyse the result of env in the scenario
    def analyse_scenario(self):

        for config in self.scenario_configurations:

            current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            junit_filename = None
            if False:
                junit_filename = config.name + current_time + ".xml"
            filename = None
            if False:
                filename = config.name + current_time + ".txt"

            if not self.manager.analyze_scenario(False, filename, junit_filename):
                print(
                    "==================================================Success!==================================================\n\n")
                self.score = 'success'
                self.final_score(self.score)
                time.sleep(1)

            else:
                print(
                    "==================================================Failure!==================================================\n\n")
                self.score = 'failure'
                self.final_score(self.score)
                time.sleep(1)

    # input the final score
    def final_score(self, score):
        self.total_score = []
        self.total_score.append(self.score)
        return self.total_score

    # write the result into local txt
    def write_result(self, times):
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

        filename = self.scenario_name + '_training_result' + ".txt"

        if times == 1:
            with open(filename, "a+") as fd:

                fd.write('========== Start tarning ! '+current_time+'==========\n')
                fd.write(str(times) + ': ' + str(self.total_score) + '\n')

        else:
            with open(filename, "a+") as fd:

                fd.write(str(times) + ': ' + str(self.total_score) + '\n')




if __name__ == '__main__':
    env = CarlaEnv()
    obs = env.reset()
    env.load_scenario()
    print(obs.shape)
    done = False
    start = time.time()
    R = 0
    i = 0
    times = 1
    while True:
        i += 1
        env.render()
        env.run_scenario()

        #the method of controling the ego_vehicle
        #obs, reward, done, info = env.step(np.clip(np.random.randn(ENV_CONFIG['action_dim']), -1, 1))
        obs, reward, done, info = env.step([0.4, 0])    #just go straight
        R += reward
        print(R)

        if i>200:
            env.analyse_scenario()
            env.write_result(times)
            env.reset()
            time.sleep(2)
            env.load_scenario()
            i = 0
            times += 1

    print(env.actor_list)
    for a in env.actor_list:
        print(a.is_alive)
        a.destroy()
    print("{:.2f} fps".format(float(i / (time.time() - start))))
