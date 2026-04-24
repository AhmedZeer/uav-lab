from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import uav_lab_1.scenes as uav_scenes
from uav_lab_1.robots.fixedwing_1 import FIXEDWING_1_CONFIG


@configclass
class UavDogfightEnvCfg(DirectMARLEnvCfg):
    """Two fixed-wing UAVs trying to reach pre-lock pursuit geometry."""

    # env
    decimation = 4
    episode_length_s = 50.0
    possible_agents = ["uav_0", "uav_1"]
    action_spaces = {"uav_0": 3, "uav_1": 3}
    observation_spaces = {"uav_0": 25, "uav_1": 25}
    state_space = 50

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=500.0, replicate_physics=True)
    ground_usd_path = os.path.join(os.path.dirname(uav_scenes.__file__), "usd", "puddles", "Puddles.usd")

    # debug visualization
    debug_follow_camera = True
    debug_follow_camera_env_id = 0
    debug_follow_camera_distance = 18.0
    debug_follow_camera_height = 6.0
    debug_follow_camera_lookahead = 12.0
    debug_follow_camera_smooth_tau = 0.15

    # robots
    uav_0_cfg: ArticulationCfg = FIXEDWING_1_CONFIG.replace(prim_path="/World/envs/env_.*/Uav0")
    uav_1_cfg: ArticulationCfg = FIXEDWING_1_CONFIG.replace(prim_path="/World/envs/env_.*/Uav1")

    # action mapping
    min_throttle = 0.35
    max_throttle = 1.0
    max_thrust = 300.0
    thrust_tau = 0.08
    max_force = 1000.0
    max_torque = 1000.0

    # reset distribution
    reset_altitude = 100.0
    reset_altitude_noise = 30.0

    reset_separation_min = 20.0
    reset_separation_max = 30.0

    reset_speed_min = 40.0
    reset_speed_max = 60.0
    reset_yaw_noise_deg = 30.0

    # observation scaling
    rel_pos_scale = 300.0
    rel_vel_scale = 50.0
    range_scale = 300.0

    # pre-lock geometry
    min_range = 3.0
    lock_range = 6.0
    desired_range = 3.5
    lock_bearing_rad = 0.25
    lock_elevation_rad = 0.20

    # safety bounds
    arena_radius = 300.0
    min_altitude = 10.0
    max_altitude = 200.0
    min_speed = 10.0
    max_speed = 400.0

    # reward scales
    rew_angle = 2.0
    rew_range = 1.2
    rew_lock_ready = 6.0
    rew_closing = 0.03
    rew_alive = 0.05
    rew_action_jerk = -0.02
    rew_tailed = -1.0
    rew_too_close = -3.0
    rew_bounds = -5.0
    rew_bad_speed = -2.0
