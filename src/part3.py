import numpy as np
import math
import json
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os


forward_length = 0.25
turn_deg = 10.0

# Simulator configuration
sim_settings = {
    "scene": "replica_v1/apartment_0/habitat/mesh_semantic.ply",
    "default_agent": 0,
    "sensor_height": 1.0,
    "width": 512,
    "height": 512,
    "sensor_pitch": 0,
}


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    return ((image / 10) * 255).astype(np.uint8)

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    return cv2.cvtColor(np.array(semantic_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # RGB, Depth, Semantic sensors
    sensors = []
    for sensor_type, uuid in [(habitat_sim.SensorType.COLOR, "color_sensor"), 
                              (habitat_sim.SensorType.DEPTH, "depth_sensor"), 
                              (habitat_sim.SensorType.SEMANTIC, "semantic_sensor")]:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = uuid
        sensor_spec.sensor_type = sensor_type
        sensor_spec.resolution = [settings["height"], settings["width"]]
        sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensors.append(sensor_spec)
    
    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=forward_length)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=turn_deg)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=turn_deg)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def navigate_and_see(sim, action, action_names, frame_idx, id_to_label, target_semantic_id, obj):
    if action not in action_names:
        return
    
    observations = sim.step(action)
    rgb_img = transform_rgb_bgr(observations["color_sensor"])

    semantic_id = id_to_label[observations["semantic_sensor"]]
    target_id_region = np.where(semantic_id == target_semantic_id)
    
    if target_id_region[0].size > 0:  
        rgb_img[target_id_region] = cv2.addWeighted(
            rgb_img[target_id_region], 0.5, np.full_like(rgb_img[target_id_region], [0, 0, 255], dtype=np.uint8), 0.5, 0
        )

    output_path = os.path.join(f"{obj}_path", f"RGB_{frame_idx}.jpg")
    cv2.imwrite(output_path, rgb_img)


def rotate(agent, direction):
    sensor_state = agent.get_state().sensor_states['color_sensor']
    cur_yaw = math.atan2(2.0 * (sensor_state.rotation.w * sensor_state.rotation.y + sensor_state.rotation.x * sensor_state.rotation.z), 
                         1.0 - 2.0 * (sensor_state.rotation.y**2 + sensor_state.rotation.z**2))
    target_yaw = -math.atan2(direction[0], -direction[1])
    yaw_diff = (target_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
    deg_diff = np.degrees(yaw_diff)
    action = 'turn_left' if deg_diff > 0 else 'turn_right'
    return abs(deg_diff), action


def load_semantic_annotations(file_path):
    with open(file_path, "r") as f:
        annotations = json.load(f)
    id_to_label = np.where(np.array(annotations["id_to_label"]) < 0, 0, annotations["id_to_label"])
    return id_to_label


def setup_simulation(sim_settings):
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    return sim, cfg


def setup_agent(sim, sim_settings, start_position):
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([start_position[0], 0.0, start_position[1]])
    agent.set_state(agent_state)
    return agent


def calculate_direction(sensor_state, target_position):
    diff = np.array([target_position[0] - sensor_state.position[0], target_position[1] - sensor_state.position[2]])
    direction = diff / np.linalg.norm(diff)
    return direction, np.linalg.norm(diff)


def rotate_agent(agent, direction, sim, action_names, frame_idx, id_to_label, target_semantic_id, obj):
    deg_diff, action = rotate(agent, direction)
    print(f'{action}: {deg_diff}')

    while deg_diff > 0:
        navigate_and_see(sim, action, action_names, frame_idx, id_to_label, target_semantic_id, obj)
        deg_diff -= turn_deg
        frame_idx += 1
    
    return frame_idx


def move_forward(agent, sim, action_names, dir_length, frame_idx, id_to_label, target_semantic_id, obj):
    while dir_length > 0:
        navigate_and_see(sim, 'move_forward', action_names, frame_idx, id_to_label, target_semantic_id, obj)
        dir_length -= forward_length
        frame_idx += 1

    return frame_idx


def navigation(path_3d, obj_points_3d, obj, target_semantic_id):
    sim, cfg = setup_simulation(sim_settings)
    agent = setup_agent(sim, sim_settings, path_3d[-1])

    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space:", action_names)

    id_to_label = load_semantic_annotations('replica_v1/apartment_0/habitat/info_semantic.json')

    if not os.path.exists(f"{obj}_path"):
        os.mkdir(f"{obj}_path")

    frame_idx = 0
    for n in reversed(range(len(path_3d))):
        sensor_state = agent.get_state().sensor_states['color_sensor']
        target_position = obj_points_3d if n == 0 else path_3d[n-1]
        direction, dir_length = calculate_direction(sensor_state, target_position)

        frame_idx = rotate_agent(agent, direction, sim, action_names, frame_idx, id_to_label, target_semantic_id, obj)

        print(f'Forward: {dir_length}')
        frame_idx = move_forward(agent, sim, action_names, dir_length, frame_idx, id_to_label, target_semantic_id, obj)