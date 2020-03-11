import habitat_sim
import cv2

import random
#%matplotlib inline
import matplotlib.pyplot as plt
import time
import numpy as np
import ipdb
st = ipdb.set_trace
import os 
import sys
import pickle
import json
#test_scene = "/hdd/matterport3d/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"

# def get_obj_id_to_obj_info_map(object_json_path):
#     objects = json.load(open(object_json_path))['objects']
#     object_id_to_obj_map = {}
#     for obj in objects:
#         object_id_to_obj_map[obj['id']] = obj 
#     return object_id_to_obj_map


mapname = "ULsKaCPVFJR"
# test_scene = "/hdd/replica/Replica-Dataset/out/{}/habitat/mesh_semantic.ply".format(mapname)
test_scene = "/hdd/matterport3d/mp3d/{}/{}.glb".format(mapname, mapname)
object_json = "/hdd/replica/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)

# object_id_to_obj_map = get_obj_id_to_obj_info_map(object_json)
# ignore_classes = ['base-cabinet','bathtub','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
take_classes = ['chair', 'lighting', 'table', 'cabinet', 'plant', 'sink', 'stool', 'sofa', 'toilet', 'tv', 'tv_monitor']

sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": True,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
SAVE = "o"
UP = "u"
DOWN = "l"
QUIT = "q"
SHOW = "s"

def make_cfg(settings):
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
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "look_up":habitat_sim.ActionSpec(
            "look_up", habitat_sim.ActuationSpec(amount=10.0)
        ),
        "look_down":habitat_sim.ActionSpec(
            "look_down", habitat_sim.ActuationSpec(amount=10.0)
        )
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([1.5, 1.072447, 0.0])
#agent_state.position = np.array([1.0, 3.0, 1.0])
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

def display_sample(rgb_obs, semantic_obs, depth_obs, visualize=False):
    unique_obs = {}
    for obj in sim.semantic_scene.objects:
        if obj.category.name()  not in unique_obs:
            unique_obs[obj.category.name()] = 1
    print("Unique objs are: ")
    print(unique_obs.keys())
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


object_id_to_obj_map = {int(obj.id.split("_")[-1]): obj for obj in sim.semantic_scene.objects}

def save_datapoint(agent, observations, data_path, viewnum):
    print("Print Sensor States.",agent.state.sensor_states)
    rgb = observations["color_sensor"]
    semantic = observations["semantic_sensor"]
    
    # Extract objects from instance segmentation
    object_list = []
    obj_ids = np.unique(semantic[70:185, 70:185])
    print("Unique semantic ids: ", obj_ids)
    # st()
    for obj_id in obj_ids:
        if obj_id not in object_id_to_obj_map:
            continue
        try:
            class_name = object_id_to_obj_map[obj_id].category.name()
        except Exception as e:
            print(e)
            st()
            print("done")
        if class_name in take_classes:
            obj_instance = object_id_to_obj_map[obj_id]
            print("Object name {}, Object category id {}, Object instance id {}".format(obj_instance.category.name(), obj_instance.category.index(), obj_id))

            obj_data = {'instance_id': obj_id, 'category_id': obj_instance.category.index(), 'category_name': obj_instance.category.name(), 'bbox_center': obj_instance.aabb.center, 'bbox_size': obj_instance.aabb.sizes}
            object_list.append(obj_data)

    # st()
    depth = observations["depth_sensor"]
    display_sample(rgb, semantic, depth, visualize=True)
    agent_pos = agent.state.position
    agent_rot = agent.state.rotation
    # Assuming all sensors have same extrinsics
    color_sensor_pos = agent.state.sensor_states['color_sensor'].position
    color_sensor_rot = agent.state.sensor_states['color_sensor'].rotation
    save_data = {'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
    return save_data
    




total_frames = 0
action_names = list(
    cfg.agents[
        sim_settings["default_agent"]
    ].action_space.keys()
)

max_frames = 1000000
plt.figure(figsize=(12 ,8))
start_flag = 0
num_saved = 0
total_views_per_scene = 6 # Keep first 3 to be camR (0 elevation)

data_folder = None
data_path = None
basepath = "/hdd/habitat_scenes_data"

cat = [obj.category for obj in sim.semantic_scene.objects if obj!=None]
idx = [c.index() for c in cat if c!=None]
# st()
while total_frames < max_frames:

    action = "move_forward"

    if(start_flag == 0):
        start_flag = 1
        observations = sim.step(action)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        display_sample(rgb, semantic, depth)

    
    keystroke = cv2.waitKey(0)
    print("keystroke: ", keystroke)

    if( 255!=keystroke and keystroke!=(-1) ):  


        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = "turn_right"
            print("action: FINISH")
            exit(1)
        elif keystroke == ord(SAVE):
            action = "save_data"
            print("action: SAVE")
        elif keystroke == ord(UP):
            action = "look_up"
            print("action: look up")
        elif keystroke == ord(DOWN):
            action = "look_down"
            print("action: look down")
        elif keystroke == ord(SHOW):
            action = "display_image"
            print("action: display_image")

        else:
            print("INVALID KEY")
            continue
        


        
        print("action", action)
        if action != "save_data" and action != "display_image":
            print("Performing action")
            observations = sim.step(action)
            rgb = observations["color_sensor"]
            semantic = observations["semantic_sensor"]
            depth = observations["depth_sensor"]
            display_sample(rgb, semantic, depth)
        else:
            
            # st()
            viewnum = str(num_saved%total_views_per_scene)
            save_data = save_datapoint(agent, observations, data_path, viewnum)
            if action == "save_data":
                if num_saved % total_views_per_scene == 0:
                    # Create new directory
                    data_folder = str(int(time.time()))
                    data_path = os.path.join(basepath, data_folder)
                    os.mkdir(data_path)
                with open(os.path.join(data_path, viewnum + ".p"), 'wb') as f:
                    pickle.dump(save_data, f)
                num_saved += 1

        
        total_frames += 1