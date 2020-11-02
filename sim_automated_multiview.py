import habitat_sim
import habitat
from habitat.config.default import get_config
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

import cv2

import random
#%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(12 ,8))
import time
import numpy as np
import ipdb
st = ipdb.set_trace
import os 
import sys
import pickle
import json


class AutomatedMultiview():
    def __init__(self):
        mapname = "apartment_0/"
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
        self.DONOTHING_KEY="r"
        self.FORWARD_KEY="w"
        self.LEFT_KEY="a"
        self.RIGHT_KEY="d"
        self.FINISH="f"
        self.SAVE = "o"
        self.UP = "u"
        self.DOWN = "l"
        self.QUIT = "q"

        self.basepath = "/hdd/habitat_scenes_data_automated"
        self.ignore_classes = ['book','base-cabinet','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
        self.rot_interval = 20.
        self.distance_thresh = 2.0
        self.num_flat_views = 3
        self.num_any_views = 7
        # self.env = habitat.Env(config=config, dataset=None)
        cfg = self.make_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(cfg)
        random.seed(self.sim_settings["seed"])
        self.sim.seed(self.sim_settings["seed"])
        self.set_agent(self.sim_settings)
        self.nav_pts = self.get_navigable_points()
        
        # self.test_navigable_points()
        self.run()

    def set_agent(self, sim_settings):
        agent = self.sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([1.5, 1.072447, 0.0])
        #agent_state.position = np.array([1.0, 3.0, 1.0])
        agent.set_state(agent_state)
        agent_state = agent.get_state()
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

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    

    def display_sample(self, rgb_obs, semantic_obs, depth_obs, visualize=False):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        # st()
        
        
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

        display_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)

        #display_img = cv2.
        cv2.imshow('img',display_img)
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
            plt.pause(0.5)
            # cv2.imshow()
            plt.close()


    def save_datapoint(self, agent, observations, data_path, viewnum, mainobj_id, flat_view):
        
        print("Print Sensor States.", self.agent.state.sensor_states)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        
        # Extract objects from instance segmentation
        object_list = []
        obj_ids = np.unique(semantic[30:230, 30:230])
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
                print("Class name is : ", class_name)
            except Exception as e:
                print(e)
                st()
                print("done")
            if class_name not in self.ignore_classes:
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


    def is_valid_datapoint(self, observations, mainobj):
        main_id = int(mainobj.id[1:])
        semantic = observations["semantic_sensor"]
        # st()
        num_occ_pixels = np.where(semantic == main_id)[0].shape[0]
        if num_occ_pixels > 1000:
            return True
        return False

    def run(self):

        scene = self.sim.semantic_scene
        objects = scene.objects
        for obj in objects:
            if obj == None or obj.category == None or obj.category.name() in self.ignore_classes:
                continue
            # st()
            print(f"Object name is: {obj.category.name()}")
            obj_center = obj.obb.to_aabb().center
            obj_center = np.expand_dims(obj_center, axis=0)
            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))
            # st()

            closest_navpts = self.nav_pts[np.where(distances<self.distance_thresh)]
            action = "do_nothing"
            flat_views = [] #flatview corresponds to moveup=100
            any_views = []

            for closest_navpt in closest_navpts[:15]:
                print("Launch position is: ", closest_navpt)
                agent_state = habitat_sim.AgentState()
                agent_state.position = closest_navpt
                self.agent.set_state(agent_state)
                self.sim.step(action)
                
                observations = self.sim.step("look_down_init")

                for moveup in range(0, 160, int(self.rot_interval)):
                    actionup = "do_nothing" if moveup==0 else "look_up"
                    print(f"actionup is: {actionup}. Moveup value is: {moveup}")
                    observations = self.sim.step(actionup)

                    for moveleft in range(0, 360, int(self.rot_interval)):
                        actionleft = "do_nothing" if moveleft==0 else "turn_left"
                        print(f"actionleft is {actionleft}")
                        observations = self.sim.step(actionleft)
                        if self.is_valid_datapoint(observations, obj):
                            if moveup == 100:
                                flat_views.append(observations)
                            else:
                                any_views.append(observations)
                        print("agent_state: position", self.agent.state.position, "rotation", self.agent.state.rotation)
                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]
                        # self.display_sample(rgb, semantic, depth)
                        # cv2.waitKey(0)
            # st()
            if len(flat_views) >= self.num_flat_views and len(any_views) >= self.num_any_views:
                data_folder = obj.category.name() + '_' + obj.id
                data_path = os.path.join(self.basepath, data_folder)
                os.mkdir(data_path)
                flat_obs = np.random.choice(flat_views, self.num_flat_views, replace=False)
                any_obs = np.random.choice(any_views, self.num_any_views, replace=False)
                viewnum = 0
                for obs in flat_obs:
                    self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, True)
                    viewnum += 1
                
                for obs in any_obs:
                    self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, False)
                    viewnum += 1

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
