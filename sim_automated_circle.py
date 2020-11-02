import habitat_sim
import habitat
from habitat.config.default import get_config
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis
import cv2
import math
import random
#%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(12 ,8))
import time
import numpy as np
import quaternion
import ipdb
st = ipdb.set_trace
import os 
import sys
import pickle
import json
from habitat_sim.utils import common as utils
# from habitat_sim.utils import viz_utils as vut
from scipy.spatial.transform import Rotation
EPSILON = 1e-8

'''
File directory in dome: /hdd/shamit/habitat/habitat-lab/HabitatScripts/sim_automated_multiview.py
conda env in dome: habitat
Directories to save files: hdd somewhere

1. change single-map selection to a loop of multiple maps(https://github.com/shamitlal/HabitatScripts/blob/master/sim_automated_multiview.py#L25)
2. check classes to keep/remove (https://github.com/shamitlal/HabitatScripts/blob/master/sim_automated_multiview.py#L50)
3. change/abandon the valid datapoint check (https://github.com/shamitlal/HabitatScripts/blob/master/sim_automated_multiview.py#L221)
4. (DONE) double-threshold to make sure our agent is not too close and not too far (https://github.com/shamitlal/HabitatScripts/blob/master/sim_automated_multiview.py#L244)
'''


class AutomatedMultiview():
    def __init__(self):

        # self.DONOTHING_KEY="r"
        # self.FORWARD_KEY="w"
        # self.LEFT_KEY="a"
        # self.RIGHT_KEY="d"
        # self.FINISH="f"
        # self.SAVE = "o"
        # self.UP = "u"
        # self.DOWN = "l"
        # self.QUIT = "q"
        self.num_episodes = 50
        self.visualize = False
        self.verbose = False

        self.mapnames = os.listdir('/hdd/replica/Replica-Dataset/out/')

        #self.ignore_classes = ['book','base-cabinet','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
        self.include_classes = ['chair', 'bed', 'toilet', 'sofa', 'indoor-plant', 'bottle', 'clock', 'refrigerator', 'tv-screen', 'vase']
        self.small_classes = ['indoor-plant', 'bottle', 'clock', 'vase']
        self.rot_interval = 20.
        self.radius_max = 3
        self.radius_min = 1
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25
        # self.env = habitat.Env(config=config, dataset=None)

        # self.test_navigable_points()
        self.run_episodes()

    def run_episodes(self):

        for episode in range(self.num_episodes):
            print("STARTING EPISODE ", episode)
            mapname = np.random.choice(self.mapnames)
            #mapname = 'apartment_0'
            self.test_scene = "/hdd/replica/Replica-Dataset/out/{}/habitat/mesh_semantic.ply".format(mapname)
            self.object_json = "/hdd/replica/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)
            self.sim_settings = {
                "width": 256,  # Spatial resolution of the observations
                "height": 256,
                "scene": self.test_scene,  # Scene path
                "default_agent": 0,
                "sensor_height": 1.5,  # Height of sensors in meters
                "color_sensor": True,  # RGB sensor
                "semantic_sensor": True,  # Semantic sensor
                "depth_sensor": True,  # Depth sensor
                "seed": 1,
            }

            self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.cfg, self.sim_cfg = self.make_cfg(self.sim_settings)
            self.sim = habitat_sim.Simulator(self.cfg)
            random.seed(self.sim_settings["seed"])
            self.sim.seed(self.sim_settings["seed"])
            self.set_agent(self.sim_settings)
            self.nav_pts = self.get_navigable_points()

            config = get_config()
            config.defrost()
            config.TASK.SENSORS = []
            config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
            config.freeze()

            self.run()
            
            self.sim.close()
            time.sleep(3)


    def set_agent(self, sim_settings):
        agent = self.sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([1.5, 1.072447, 0.0])
        #agent_state.position = np.array([1.0, 3.0, 1.0])
        agent.set_state(agent_state)
        agent_state = agent.get_state()
        if self.verbose:
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        self.agent = agent

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings["scene"]

        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "do_nothing": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.)
            ),
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
            ),
            "look_up":habitat_sim.ActionSpec(
                "look_up", habitat_sim.ActuationSpec(amount=self.rot_interval)
            ),
            "look_down":habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=self.rot_interval)
            ),
            "look_down_init":habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=100.0)
            )
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg]), sim_cfg 
    

    def display_sample(self, rgb_obs, semantic_obs, depth_obs, mainobj=None, visualize=False):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        # st()
        
        
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

        display_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)
        #print(display_img.shape)

        # mask_image = False
        # if mask_image and mainobj is not None:
        #     main_id = int(mainobj.id[1:])
        #     print("MAINID ", main_id)
        #     # semantic = observations["semantic_sensor"]
        #     display_img[semantic_obs == main_id] = [1, 0, 1]
            # st()

        #display_img = cv2
        plt.imshow(display_img)
        plt.show()
        # cv2.imshow('img',display_img)
        if visualize:
            arr = [rgb_img, semantic_img, depth_img]
            titles = ['rgb', 'semantic', 'depth']
            plt.figure(figsize=(12 ,8))
            for i, data in enumerate(arr):
                ax = plt.subplot(1, 3, i+1)
                ax.axis('off')
                ax.set_title(titles[i])
                plt.imshow(data)
                # plt.pause()
            plt.show()
            # plt.pause(0.5)
            # cv2.imshow()
            # plt.close()


    def save_datapoint(self, agent, observations, data_path, viewnum, mainobj_id, flat_view):
        if self.verbose:
            print("Print Sensor States.", self.agent.state.sensor_states)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        
        # Extract objects from instance segmentation
        object_list = []
        obj_ids = np.unique(semantic[30:230, 30:230])
        if self.verbose:
            print("Unique semantic ids: ", obj_ids)

        # st()
        for obj_id in obj_ids:
            if obj_id < 1 or obj_id > len(self.sim.semantic_scene.objects):
                continue
            if self.sim.semantic_scene.objects[obj_id] == None:
                continue
            if self.sim.semantic_scene.objects[obj_id].category == None:
                continue
            try:
                class_name = self.sim.semantic_scene.objects[obj_id].category.name()
                if self.verbose:
                    print("Class name is : ", class_name)
            except Exception as e:
                print(e)
                st()
                print("done")
            #if class_name not in self.ignore_classes:
            if class_name in self.include_classes:
                obj_instance = self.sim.semantic_scene.objects[obj_id]
                # print("Object name {}, Object category id {}, Object instance id {}".format(class_name, obj_instance['id'], obj_instance['class_id']))

                obj_data = {'instance_id': obj_id, 'category_id': obj_instance.category.index(), 'category_name': obj_instance.category.name(), 'bbox_center': obj_instance.obb.to_aabb().center, 'bbox_size': obj_instance.obb.to_aabb().sizes}
                # object_list.append(obj_instance)
                object_list.append(obj_data)

        # st()
        depth = observations["depth_sensor"]
        # self.display_sample(rgb, semantic, depth, visualize=True)
        agent_pos = agent.state.position
        agent_rot = agent.state.rotation
        # Assuming all sensors have same extrinsics
        color_sensor_pos = agent.state.sensor_states['color_sensor'].position
        color_sensor_rot = agent.state.sensor_states['color_sensor'].rotation
        save_data = {'flat_view': flat_view, 'mainobj_id': mainobj_id, 'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
        
        with open(os.path.join(data_path, str(viewnum) + ".p"), 'wb') as f:
            pickle.dump(save_data, f)
        f.close()


    def is_valid_datapoint(self, observations, mainobj):
        main_id = int(mainobj.id[1:])
        semantic = observations["semantic_sensor"]
        # st()
        num_occ_pixels = np.where(semantic == main_id)[0].shape[0]
        #print(semantic.shape)
        #print("Number of pixels: ", num_occ_pixels)
        small_objects = []
        if mainobj.category.name() in self.small_classes:
            if num_occ_pixels > 30 and num_occ_pixels < 0.75*256*256:
                return True
        else:
            if num_occ_pixels > 300 and num_occ_pixels < 0.75*256*256: 
                return True
        return False

    def quaternion_from_two_vectors(self, v0: np.array, v1: np.array) -> np.quaternion:
        r"""Computes the quaternion representation of v1 using v0 as the origin."""
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        c = v0.dot(v1)
        # Epsilon prevents issues at poles.
        if c < (-1 + EPSILON):
            c = max(c, -1)
            m = np.stack([v0, v1], 0)
            _, _, vh = np.linalg.svd(m, full_matrices=True)
            axis = vh.T[:, 2]
            w2 = (1 + c) * 0.5
            w = np.sqrt(w2)
            axis = axis * np.sqrt(1 - w2)
            return np.quaternion(w, *axis)

        axis = np.cross(v0, v1)
        s = np.sqrt((1 + c) * 2)
        return np.quaternion(s * 0.5, *(axis / s))

    def run(self):

        scene = self.sim.semantic_scene
        objects = scene.objects
        for obj in objects:
            #if obj == None or obj.category == None or obj.category.name() in self.ignore_classes:
            if obj == None or obj.category == None or obj.category.name() not in self.include_classes:
                continue
            # st()
            if self.verbose:
                print(f"Object name is: {obj.category.name()}")
            # Calculate distance to object center
            obj_center = obj.obb.to_aabb().center
            #print(obj_center)
            obj_center = np.expand_dims(obj_center, axis=0)
            #print(obj_center)
            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

            # Get points with r_min < dist < r_max
            valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]
            # if not valid_pts:
                # continue

            # plot valid points that we happen to select
            # self.plot_navigable_points(valid_pts)

            # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
            valid_pts_shift = valid_pts - obj_center
            # valid_pts_flat = valid_pts.copy()
            # valid_pts_flat[:, 1] = 0.0

            # obj_center_flat = obj_center.copy()
            # obj_center_flat[:, 1] = 0.0
            
            # st()
            #valid_quats = np.vstack([quaternion.as_float_array(quat_from_two_vectors(obj_center[0], valid_pt)) for valid_pt in valid_pts])
            #valid_eulers = Rotation.from_quat(valid_quats).as_euler('xyz', degrees=True) + 180
            #valid_yaw = valid_eulers[:, 2]
            #valid_pitch = np.degrees(np.arctan((np.sqrt(valid_pts_shift[:,0]**2+valid_pts_shift[:,2]**2)/valid_pts_shift[:,1]**2))) #(phi) (y)
            #valid_yaw = np.degrees(np.arccos(valid_pts_shift[:,0]/(np.sqrt(valid_pts_shift[:,0]**2+valid_pts_shift[:,1]**2)))) # (theta) (z)
            valid_yaw = np.degrees(np.arctan2(valid_pts_shift[:,2],valid_pts_shift[:,0]))


            nbins = 18
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size

            spawns_per_bin = int(self.num_views / num_valid_bins) + 2
            print(f'spawns_per_bin: {spawns_per_bin}')

            if self.visualize:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                plt.show()

            # 1. angle calculation: https://math.stackexchange.com/questions/2521886/how-to-find-angle-between-2-points-in-3d-space
            # 2. angle quantization (round each angle to the nearst bin, 163 --> 160, 47 --> 50, etc.) - STILL NEED

            '''
            # 3. loop around the sphere and get views
            for vert_angle in vertical_angle:
                for hori_angle in horizontal_angle:
                    # for i in range(num_view_per_angle)
                    valid_pts_bin[vert_angle, hori_angle][np.random.choice()]
                    # spawn agent
                    # rotate the agent with our calculated angle (so that the agent looks at the object)
                    # if is_valid_datapoint: save it to views
                    # self.display_sample(rgb, semantic, depth)
            '''

            action = "do_nothing"
            # flat_views = [] #flatview corresponds to moveup=100
            # any_views = []
            episodes = []
            # spawns_per_bin = 2
            valid_pts_selected = []
            for b in range(nbins):
                
                # get all angle indices in the current bin range
                # st()
                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                if inds_bin_cur[0].size == 0:
                    continue

                for s in range(spawns_per_bin):
                    # st()
                    s_ind = np.random.choice(inds_bin_cur[0])
                    pos_s = valid_pts[s_ind]
                    valid_pts_selected.append(pos_s)
                    agent_state = habitat_sim.AgentState()
                    agent_state.position = pos_s

                    agent_to_obj = np.squeeze(obj_center) - agent_state.position
                    agent_local_forward = np.array([0, 0, -1.0]) # y, z, x
                    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                    flat_to_obj /= flat_dist_to_obj

                    det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                    #print("DET", det)
                    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                    #print("TURN", turn_angle)
                    #print(det)
                    quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))
                    agent_state.rotation = quat_yaw
                    # print(quat_yaw)

                    # valid_pitch = math.atan2(np.sqrt(agent_to_obj[2]**2+agent_to_obj[1]**2),agent_to_obj[0])
                    # #valid_pitch = math.atan2(agent_to_obj[1],agent_to_obj[2])

                    # quat_pitch = quat_from_angle_axis(valid_pitch, np.array([1.0, 0, 0]))
                    # print(quat_pitch)

                    # print(quat_pitch + quat_yaw)

                    # quat_comb = habitat.utils.geometry_utils.quaternion_from_two_vectors(np.squeeze(obj_center), agent_state.position)

                    # agent_state.rotation = quat_comb

                    # print(agent_state.rotation)

                    ############
                    # st()
                    # get quaternion from euler angles
                    # rot = Rotation.from_euler('xyz', [valid_yaw[s_ind], 0, valid_pitch[s_ind]], degrees=True)
                    # rot_quat = np.quaternion(*(rot.as_quat()))

                    # # try
                    # agent_local_forward = np.array([0, 0, -1.0])
                    # flat_to_obj = np.array([valid_pts_shift[s_ind, 0], 0.0, valid_pts_shift[s_ind, 2]])
                    # flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                    # flat_to_obj /= flat_dist_to_obj

                    # det = (
                    #     flat_to_obj[0] * agent_local_forward[2]
                    #     - agent_local_forward[0] * flat_to_obj[2]
                    # )
                    # turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                    # rot = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

                    #######
                    # print("OBJECT CENTER: ", obj_center)
                    # print("POINT: ", valid_pts[s_ind])
                    # rot_s = quat_from_two_vectors(np.squeeze(obj_center), pos_s)

                    #self.cfg.freeze()
                    # with habitat.Env(config=self.sim_cfg, dataset=None) as env:
                    #     observations = self.agent.get_observations_at(list(pos_s), list(rot_s))
                    # print(observations)
                
                    # = rot
                    
                    # change sensor state to default 
                    # need to move the sensors too
                    for sensor in agent_state.sensor_states:
                        agent_state.sensor_states[sensor].rotation = agent_state.rotation
                        agent_state.sensor_states[sensor].position = agent_state.position

                    # agent_state.rotation = rot
                    
                    self.agent.set_state(agent_state)
                    observations = self.sim.step(action)
                    
                    if self.is_valid_datapoint(observations, obj):
                        if self.verbose:
                            print("episode is valid......")
                        episodes.append(observations)
                        if self.visualize:
                            rgb = observations["color_sensor"]
                            semantic = observations["semantic_sensor"]
                            depth = observations["depth_sensor"]
                            self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)
                    # if self.visualize:
                    #         rgb = observations["color_sensor"]
                    #         semantic = observations["semantic_sensor"]
                    #         depth = observations["depth_sensor"]
                    #         self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)

                    #print("agent_state: position", self.agent.state.position, "rotation", self.agent.state.rotation)
                    

                    # Visualize
                    
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        
            
        #     vut.make_video(
        #     observations=episodes,
        #     primary_obs="color_sensor",
        #     primary_obs_type="color",
        #     video_file= './' + "rotation",
        #     fps=3,
        #     open_vid=self.visualize,
        # )

            if len(episodes) >= self.num_views:
                print(f'num episodes: {len(episodes)}')
                data_folder = obj.category.name() + '_' + obj.id
                data_path = os.path.join(self.basepath, data_folder)
                print("Saving to ", data_path)
                os.mkdir(data_path)
                flat_obs = np.random.choice(episodes, self.num_views, replace=False)
                viewnum = 0
                for obs in flat_obs:
                    self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, True)
                    viewnum += 1
            else:
                print(f"Not enough episodes: f{len(episodes)}")

            if self.visualize:
                valid_pts_selected = np.vstack(valid_pts_selected)
                self.plot_navigable_points(valid_pts_selected)

            # data_folder = obj.category.name() + '_' + obj.id
            # data_path = os.path.join(self.basepath, data_folder)
            # for obs in episodes:
            #     self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, False)
            #     viewnum += 1


                            # for obs in any_obs:
                #     self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, False)
                #     viewnum += 1
                # self.display_sample(rgb, semantic, depth)
                # cv2.waitKey(0)


            # action = "do_nothing"
            # flat_views = [] #flatview corresponds to moveup=100
            # any_views = []

            # for closest_navpt in closest_navpts[:15]:
            #     print("Launch position is: ", closest_navpt)
            #     agent_state = habitat_sim.AgentState()
            #     agent_state.position = closest_navpt
            #     self.agent.set_state(agent_state)
            #     self.sim.step(action)
                
            #     observations = self.sim.step("look_down_init")

            #     for moveup in range(0, 160, int(self.rot_interval)):
            #         actionup = "do_nothing" if moveup==0 else "look_up"
            #         print(f"actionup is: {actionup}. Moveup value is: {moveup}")
            #         observations = self.sim.step(actionup)

            #         for moveleft in range(0, 360, int(self.rot_interval)):
            #             actionleft = "do_nothing" if moveleft==0 else "turn_left"
            #             print(f"actionleft is {actionleft}")
            #             observations = self.sim.step(actionleft)
            #             if self.is_valid_datapoint(observations, obj):
            #                 if moveup == 100:
            #                     flat_views.append(observations)
            #                 else:
            #                     any_views.append(observations)
            #             print("agent_state: position", self.agent.state.position, "rotation", self.agent.state.rotation)
            #             rgb = observations["color_sensor"]
            #             semantic = observations["semantic_sensor"]
            #             depth = observations["depth_sensor"]
            #             # self.display_sample(rgb, semantic, depth)
            #             # cv2.waitKey(0)
            # st()
            # if len(flat_views) >= self.num_flat_views and len(any_views) >= self.num_any_views:
            #     data_folder = obj.category.name() + '_' + obj.id
            #     data_path = os.path.join(self.basepath, data_folder)
            #     os.mkdir(data_path)
            #     flat_obs = np.random.choice(flat_views, self.num_flat_views, replace=False)
            #     any_obs = np.random.choice(any_views, self.num_any_views, replace=False)
            #     viewnum = 0
            #     for obs in flat_obs:
            #         self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, True)
            #         viewnum += 1
                
                # for obs in any_obs:
                #     self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, False)
                #     viewnum += 1

    def get_navigable_points(self):
        navigable_points = np.array([0,0,0])
        for i in range(20000):
            navigable_points = np.vstack((navigable_points,self.sim.pathfinder.get_random_navigable_point()))
        return navigable_points
    
    def plot_navigable_points(self, points):
        # print(points)
        x_sample = points[:,0]
        z_sample = points[:,2]
        plt.plot(z_sample, x_sample, 'o', color = 'red')
        
        plt.show()

    # # Just visualize the scene from different navigable points
    # def test_navigable_points(self):
        
    #     action = "move_forward"
    #     for pts in self.nav_pts:
    #         keystroke = cv2.waitKey(0)
    #         print("keystroke: ", keystroke)
    #         print("Launch position is: ", pts)
    #         agent_state = habitat_sim.AgentState()
    #         agent_state.position = pts
    #         self.agent.set_state(agent_state)
    #         observations = self.sim.step(action)
    #         rgb = observations["color_sensor"]
    #         semantic = observations["semantic_sensor"]
    #         depth = observations["depth_sensor"]
    #         self.display_sample(rgb, semantic, depth)

    # # Test whether we actually spawn near given object when we select navigable point near it.
    # def test_navigable_point_for_single_obj(self):
    #     scene = self.sim.semantic_scene
    #     objects = scene.objects
    #     for obj in objects:
    #         if obj == None or obj.category == None:
    #             continue
    #         print(f"Object name is: {obj.category.name()}")

    #         if obj.category.name() == "bathtub":
    #             obj_center = obj.obb.to_aabb().center
    #             obj_center = np.expand_dims(obj_center, axis=0)
    #             distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))
    #             closest_navpt = self.nav_pts[distances.argmin()]
    #             action = "move_forward"
    #             print("Launch position is: ", closest_navpt)
    #             agent_state = habitat_sim.AgentState()
    #             agent_state.position = closest_navpt
    #             self.agent.set_state(agent_state)
    #             while True:
                    
    #                 observations = self.sim.step(action)
    #                 rgb = observations["color_sensor"]
    #                 semantic = observations["semantic_sensor"]
    #                 depth = observations["depth_sensor"]
    #                 self.display_sample(rgb, semantic, depth)
    #                 keystroke = cv2.waitKey(0)
    #                 print("keystroke: ", keystroke)

    #                 if( 255!=keystroke and keystroke!=(-1) ):  
                        
    #                     if keystroke == ord(self.DONOTHING_KEY):
    #                         action = "do_nothing"
    #                         print("action: DONOTHING")
    #                     elif keystroke == ord(self.FORWARD_KEY):
    #                         action = "move_forward"
    #                         print("action: FORWARD")
    #                     elif keystroke == ord(self.LEFT_KEY):
    #                         action = "turn_left"
    #                         print("action: LEFT")
    #                     elif keystroke == ord(self.RIGHT_KEY):
    #                         action = "turn_right"
    #                         print("action: RIGHT")
    #                     elif keystroke == ord(self.FINISH):
    #                         action = "turn_right"
    #                         print("action: FINISH")
    #                         exit(1)
    #                     elif keystroke == ord(self.SAVE):
    #                         action = "save_data"
    #                         print("action: SAVE")
    #                     elif keystroke == ord(self.UP):
    #                         action = "look_up"
    #                         print("action: look up")
    #                     elif keystroke == ord(self.DOWN):
    #                         action = "look_down"
    #                         print("action: look down")
    #                     else:
    #                         print("INVALID KEY")
    #                         continue



if __name__ == '__main__':
    AutomatedMultiview()