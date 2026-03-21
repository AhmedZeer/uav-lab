import argparse
import csv
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Circle-flight debug scene for FixedWing-1.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--mode",
    type=str,
    default="predefined_force",
    choices=["predefined_force", "minimal_controller"],
    help="Control mode.",
)
parser.add_argument("--radius", type=float, default=80.0, help="Target circle radius [m].")
parser.add_argument("--center_x", type=float, default=0.0, help="Circle center x in world frame [m].")
parser.add_argument("--center_y", type=float, default=0.0, help="Circle center y in world frame [m].")
parser.add_argument("--clockwise", action="store_true", help="Use clockwise circle direction.")
parser.add_argument("--target_speed", type=float, default=20.0, help="Target forward speed [m/s].")
parser.add_argument("--target_alt", type=float, default=20.0, help="Target altitude [m].")
parser.add_argument("--max_thrust", type=float, default=100.0, help="Maximum thrust [N].")
parser.add_argument("--max_torque", type=float, default=10.0, help="Maximum body torque [N*m].")

# Predefined force mode params.
parser.add_argument("--predef_forward_force", type=float, default=85.0, help="Body +X force [N].")
parser.add_argument("--predef_lateral_force", type=float, default=18.0, help="Body +/-Y force [N].")

# Minimal controller gains.
parser.add_argument("--k_radial", type=float, default=0.05, help="Radial error gain.")
parser.add_argument("--k_heading", type=float, default=1.2, help="Heading-to-roll gain.")
parser.add_argument("--k_roll", type=float, default=2.0, help="Roll angle P gain.")
parser.add_argument("--k_p", type=float, default=0.35, help="Roll-rate damping gain.")
parser.add_argument("--k_alt", type=float, default=0.03, help="Altitude-to-pitch gain.")
parser.add_argument("--k_vz", type=float, default=0.08, help="Vertical speed damping gain.")
parser.add_argument("--k_pitch", type=float, default=1.8, help="Pitch angle P gain.")
parser.add_argument("--k_q", type=float, default=0.3, help="Pitch-rate damping gain.")
parser.add_argument("--k_speed", type=float, default=0.03, help="Speed-to-throttle gain.")
parser.add_argument("--thrust_trim", type=float, default=0.58, help="Throttle trim [0..1].")
parser.add_argument("--k_beta", type=float, default=0.4, help="Sideslip-to-yaw damping gain.")
parser.add_argument("--k_r", type=float, default=0.15, help="Yaw-rate damping gain.")
parser.add_argument("--max_bank_deg", type=float, default=45.0, help="Max commanded bank angle [deg].")
parser.add_argument("--max_pitch_deg", type=float, default=18.0, help="Max commanded pitch angle [deg].")

parser.add_argument("--start_alt", type=float, default=20.0, help="Initial altitude [m].")
parser.add_argument("--start_speed", type=float, default=18.0, help="Initial forward speed [m/s].")
parser.add_argument("--start_yaw_deg", type=float, default=0.0, help="Initial yaw [deg].")
parser.add_argument("--enable_aero", action="store_true", help="Include aero+drag model in predefined force mode.")

parser.add_argument("--debug_print_hz", type=float, default=5.0, help="Console print rate [Hz].")
parser.add_argument("--log_csv", action="store_true", help="Write telemetry CSV.")
parser.add_argument("--log_path", type=str, default="logs/circle_debug.csv", help="Telemetry CSV path.")
parser.add_argument("--log_hz", type=float, default=50.0, help="CSV logging rate [Hz].")
parser.add_argument("--log_env_id", type=int, default=0, help="Environment index for logging.")
parser.add_argument("--follow_cam", action="store_true", help="Enable smoothed chase camera.")
parser.add_argument("--follow_cam_env_id", type=int, default=0, help="Environment index for chase camera.")
parser.add_argument("--follow_cam_distance", type=float, default=10.0, help="Chase distance behind UAV [m].")
parser.add_argument("--follow_cam_height", type=float, default=3.0, help="Chase camera height above UAV [m].")
parser.add_argument("--follow_cam_lookahead", type=float, default=6.0, help="Look-ahead distance in front of UAV [m].")
parser.add_argument("--follow_cam_smooth_tau", type=float, default=0.35, help="Smoothing time constant for chase camera [s].")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils
from uav_lab_1.robots.aero import (
    AeroConfig,
    ControlSigns,
    DragConfig,
    PropConfig,
    calculate_propeller_thrust,
    clamp,
    compute_aero_forces_and_moments,
    linear_drag,
)
from uav_lab_1.robots.follow_camera import FollowCameraConfig, SmoothedFollowCamera
from uav_lab_1.scenes import BasicFixedWing1SceneCfg

MAX_FORCE = 1000.0
MAX_TORQUE = 1000.0

PROP_CFG = PropConfig()
CONTROL_SIGNS = ControlSigns()
DRAG_CFG = DragConfig()
AERO_CFG = AeroConfig(
    # # Softer rate derivatives for this open-loop circle-debug scene.
    # Cl_p=-0.20,
    # Cm_q=-3.0,
    # Cn_p=-0.03,
    # Cn_r=-0.10,
    # # Keep the linear model in its valid small-disturbance range.
    # alpha_limit_rad=0.55,
    # beta_limit_rad=0.45,
    # rate_hat_limit=0.20,
)


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _quat_from_yaw_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _euler_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _init_airborne_state(uav, scene: InteractiveScene, altitude: float, speed: float, yaw_deg: float):
    device = uav.device
    num_envs = uav.num_instances
    env_ids = torch.arange(num_envs, device=device, dtype=torch.int64)

    default_root_state = uav.data.default_root_state.clone()
    root_pose = default_root_state[:, :7].clone()
    root_vel = default_root_state[:, 7:].clone()

    root_pose[:, :3] = scene.env_origins.to(device) + torch.tensor([0.0, 0.0, altitude], device=device)
    yaw = torch.full((num_envs,), math.radians(yaw_deg), device=device)
    root_pose[:, 3:7] = _quat_from_yaw_wxyz(yaw)

    root_vel[:, 0] = speed * torch.cos(yaw)
    # root_vel[:, 1] = speed * torch.sin(yaw)
    root_vel[:, 2] = 0.0
    root_vel[:, 3:6] = 0.0

    uav.write_root_pose_to_sim(root_pose, env_ids)
    uav.write_root_velocity_to_sim(root_vel, env_ids)


def _thrust_force_from_cmd(thrust_cmd: torch.Tensor, max_thrust: float) -> torch.Tensor:
    thrust = clamp(thrust_cmd, 0.0, 1.0)
    thrust = torch.clamp(CONTROL_SIGNS.throttle * thrust, min=0.0, max=1.0)
    return calculate_propeller_thrust(thrust, max_thrust, PROP_CFG)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    uav = scene["uav"]
    body_ids, _ = uav.find_bodies("body")
    if len(body_ids) == 0:
        body_ids = [0]

    device = uav.device
    num_envs = uav.num_instances
    body_ids_t = torch.tensor(body_ids, device=device, dtype=torch.int32)
    env_i = int(max(0, min(num_envs - 1, int(args_cli.log_env_id)))) if num_envs > 0 else 0

    debug_interval = (1.0 / float(args_cli.debug_print_hz)) if float(args_cli.debug_print_hz) > 0.0 else None
    last_debug_t = -1.0e9
    last_debug_len = 0

    log_writer = None
    log_file = None
    log_interval = (1.0 / float(args_cli.log_hz)) if float(args_cli.log_hz) > 0.0 else 0.0
    last_log_t = -1.0e9
    if args_cli.log_csv:
        log_dir = os.path.dirname(args_cli.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(args_cli.log_path, "w", newline="", buffering=1)
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "sim_time_s",
                "mode",
                "x_w",
                "y_w",
                "z_w",
                "speed_mps",
                "u_mps",
                "v_mps",
                "w_mps",
                "p_rps",
                "q_rps",
                "r_rps",
                "roll_rad",
                "pitch_rad",
                "yaw_rad",
                "radius_err_m",
                "heading_err_rad",
                "thrust_cmd",
                "roll_cmd",
                "pitch_cmd",
                "yaw_cmd",
                "force_x",
                "force_y",
                "force_z",
                "torque_x",
                "torque_y",
                "torque_z",
            ]
        )
        print(f"[INFO]: Logging to {args_cli.log_path}")

    center = torch.tensor([float(args_cli.center_x), float(args_cli.center_y)], device=device)
    turn_sign = -1.0 if args_cli.clockwise else 1.0
    max_bank = math.radians(float(args_cli.max_bank_deg))
    max_pitch = math.radians(float(args_cli.max_pitch_deg))
    g = 9.81

    print(f"[INFO]: Running mode={args_cli.mode}, radius={args_cli.radius}m, target_speed={args_cli.target_speed}m/s")

    follow_cam = None
    if args_cli.follow_cam:
        follow_cam = SmoothedFollowCamera(
            sim=sim,
            sim_dt=sim_dt,
            device=device,
            num_envs=num_envs,
            cfg=FollowCameraConfig(
                env_id=int(args_cli.follow_cam_env_id),
                distance=float(args_cli.follow_cam_distance),
                height=float(args_cli.follow_cam_height),
                lookahead=float(args_cli.follow_cam_lookahead),
                smooth_tau=float(args_cli.follow_cam_smooth_tau),
            ),
        )
        print(
            "[INFO]: Smoothed follow camera enabled "
            f"(env {follow_cam.env_id}, tau={follow_cam.cfg.smooth_tau:.2f}s)."
        )

    try:
        while simulation_app.is_running():
            state = uav.data.root_state_w
            quat_wxyz = state[:, 3:7]
            v_world = state[:, 7:10]
            w_world = state[:, 10:13]
            pos_world = state[:, 0:3]

            v_body = math_utils.quat_apply_inverse(quat_wxyz, v_world)
            w_body = math_utils.quat_apply_inverse(quat_wxyz, w_world)
            roll, pitch, yaw = _euler_from_quat_wxyz(quat_wxyz)

            if args_cli.mode == "predefined_force":
                thrust_cmd = torch.full((num_envs,), float(args_cli.thrust_trim), device=device)
                roll_cmd = torch.zeros((num_envs,), device=device)
                pitch_cmd = torch.zeros((num_envs,), device=device)
                yaw_cmd = torch.zeros((num_envs,), device=device)

                body_force = torch.zeros((num_envs, 3), device=device)
                body_force[:, 0] = float(args_cli.predef_forward_force)
                body_force[:, 1] = turn_sign * float(args_cli.predef_lateral_force)

                control_torque = torch.zeros((num_envs, 3), device=device)

                if args_cli.enable_aero:
                    aero_forces, aero_moments = compute_aero_forces_and_moments(
                        v_body,
                        w_body,
                        torch.zeros((num_envs,), device=device),
                        torch.zeros((num_envs,), device=device),
                        torch.zeros((num_envs,), device=device),
                        AERO_CFG,
                    )
                    drag_force = linear_drag(v_body, DRAG_CFG)
                else:
                    aero_forces = torch.zeros((num_envs, 3), device=device)
                    aero_moments = torch.zeros((num_envs, 3), device=device)
                    drag_force = torch.zeros((num_envs, 3), device=device)

                total_forces = clamp(body_force + aero_forces + drag_force, -MAX_FORCE, MAX_FORCE)
                total_torques = clamp(control_torque + aero_moments, -MAX_TORQUE, MAX_TORQUE)
            else:
                pos_xy = pos_world[:, 0:2]
                rel = pos_xy - center.unsqueeze(0)
                radius = torch.linalg.norm(rel, dim=-1).clamp(min=1.0)
                radial_dir = rel / radius.unsqueeze(-1)
                tangent_dir = torch.stack([-radial_dir[:, 1], radial_dir[:, 0]], dim=-1) * turn_sign

                speed_xy = torch.linalg.norm(v_world[:, 0:2], dim=-1).clamp(min=0.1)
                course_des = torch.atan2(tangent_dir[:, 1], tangent_dir[:, 0])
                heading_err = _wrap_pi(course_des - yaw)
                radius_err = radius - float(args_cli.radius)

                phi_ff = torch.atan((speed_xy**2) / (g * max(float(args_cli.radius), 1.0))) * turn_sign
                phi_des = phi_ff + float(args_cli.k_heading) * heading_err + float(args_cli.k_radial) * radius_err
                phi_des = torch.clamp(phi_des, -max_bank, max_bank)
                roll_err = _wrap_pi(phi_des - roll)
                roll_cmd = clamp(float(args_cli.k_roll) * roll_err - float(args_cli.k_p) * w_body[:, 0], -1.0, 1.0)

                alt_err = float(args_cli.target_alt) - pos_world[:, 2]
                pitch_des = float(args_cli.k_alt) * alt_err - float(args_cli.k_vz) * v_world[:, 2]
                pitch_des = torch.clamp(pitch_des, -max_pitch, max_pitch)
                pitch_err = _wrap_pi(pitch_des - pitch)
                pitch_cmd = clamp(float(args_cli.k_pitch) * pitch_err - float(args_cli.k_q) * w_body[:, 1], -1.0, 1.0)

                speed_err = float(args_cli.target_speed) - v_body[:, 0]
                thrust_cmd = clamp(float(args_cli.thrust_trim) + float(args_cli.k_speed) * speed_err, 0.0, 1.0)

                safe_speed = torch.clamp(torch.linalg.norm(v_body, dim=-1), min=0.1)
                beta = torch.asin(torch.clamp(-v_body[:, 1] / safe_speed, min=-1.0, max=1.0))
                yaw_cmd = clamp(-float(args_cli.k_beta) * beta - float(args_cli.k_r) * w_body[:, 2], -1.0, 1.0)

                thrust_force = _thrust_force_from_cmd(thrust_cmd, float(args_cli.max_thrust))
                aileron = roll_cmd * CONTROL_SIGNS.aileron
                elevator = pitch_cmd * CONTROL_SIGNS.elevator
                rudder = yaw_cmd * CONTROL_SIGNS.rudder
                control_torque = torch.stack(
                    [aileron * float(args_cli.max_torque), elevator * float(args_cli.max_torque), rudder * float(args_cli.max_torque)],
                    dim=-1,
                )

                aero_forces, aero_moments = compute_aero_forces_and_moments(v_body, w_body, aileron, elevator, rudder, AERO_CFG)
                drag_force = linear_drag(v_body, DRAG_CFG)
                total_forces = clamp(thrust_force + aero_forces + drag_force, -MAX_FORCE, MAX_FORCE)
                total_torques = clamp(control_torque + aero_moments, -MAX_TORQUE, MAX_TORQUE)

            com_positions = torch.zeros((num_envs, len(body_ids), 3), device=device)
            com_positions[:, 0, :] = uav.data.body_com_pos_b[:, 0]

            forces = torch.zeros((num_envs, len(body_ids), 3), device=device)
            forces[:, 0, :] = total_forces
            torques = torch.zeros((num_envs, len(body_ids), 3), device=device)
            torques[:, 0, :] = total_torques

            composer = uav.instantaneous_wrench_composer
            composer.add_forces_and_torques(forces=forces, positions=com_positions, body_ids=body_ids_t)
            composer.add_forces_and_torques(torques=torques, body_ids=body_ids_t)

            if debug_interval is not None and (sim_time - last_debug_t) >= debug_interval:
                rel0 = pos_world[env_i, 0:2] - center
                rad0 = torch.linalg.norm(rel0).item()
                radius_err0 = rad0 - float(args_cli.radius)
                line = (
                    f"env{env_i} mode={args_cli.mode} pos=({pos_world[env_i,0].item():7.1f},{pos_world[env_i,1].item():7.1f},{pos_world[env_i,2].item():6.1f}) "
                    f"v=({v_body[env_i,0].item():6.2f},{v_body[env_i,1].item():6.2f},{v_body[env_i,2].item():6.2f}) "
                    f"pqr=({w_body[env_i,0].item():6.2f},{w_body[env_i,1].item():6.2f},{w_body[env_i,2].item():6.2f}) "
                    f"F=({total_forces[env_i,0].item():7.2f},{total_forces[env_i,1].item():7.2f},{total_forces[env_i,2].item():7.2f}) "
                    f"T=({total_torques[env_i,0].item():7.2f},{total_torques[env_i,1].item():7.2f},{total_torques[env_i,2].item():7.2f}) "
                    f"r_err={radius_err0:7.2f}"
                )
                pad = " " * max(0, last_debug_len - len(line))
                print(f"\r{line}{pad}", end="", flush=True)
                last_debug_t = sim_time
                last_debug_len = len(line)

            if log_writer is not None and (log_interval == 0.0 or (sim_time - last_log_t) >= log_interval):
                rel0 = pos_world[env_i, 0:2] - center
                radius0 = torch.linalg.norm(rel0).item()
                radius_err0 = radius0 - float(args_cli.radius)
                heading_err0 = 0.0
                if args_cli.mode == "minimal_controller":
                    tangent = torch.tensor([-rel0[1], rel0[0]], device=device) * turn_sign
                    course_des = torch.atan2(tangent[1], tangent[0])
                    heading_err0 = _wrap_pi(course_des - yaw[env_i]).item()

                log_writer.writerow(
                    [
                        float(sim_time),
                        args_cli.mode,
                        float(pos_world[env_i, 0].item()),
                        float(pos_world[env_i, 1].item()),
                        float(pos_world[env_i, 2].item()),
                        float(torch.linalg.norm(v_body[env_i]).item()),
                        float(v_body[env_i, 0].item()),
                        float(v_body[env_i, 1].item()),
                        float(v_body[env_i, 2].item()),
                        float(w_body[env_i, 0].item()),
                        float(w_body[env_i, 1].item()),
                        float(w_body[env_i, 2].item()),
                        float(roll[env_i].item()),
                        float(pitch[env_i].item()),
                        float(yaw[env_i].item()),
                        float(radius_err0),
                        float(heading_err0),
                        float(thrust_cmd[env_i].item() if isinstance(thrust_cmd, torch.Tensor) else thrust_cmd),
                        float(roll_cmd[env_i].item() if isinstance(roll_cmd, torch.Tensor) else roll_cmd),
                        float(pitch_cmd[env_i].item() if isinstance(pitch_cmd, torch.Tensor) else pitch_cmd),
                        float(yaw_cmd[env_i].item() if isinstance(yaw_cmd, torch.Tensor) else yaw_cmd),
                        float(total_forces[env_i, 0].item()),
                        float(total_forces[env_i, 1].item()),
                        float(total_forces[env_i, 2].item()),
                        float(total_torques[env_i, 0].item()),
                        float(total_torques[env_i, 1].item()),
                        float(total_torques[env_i, 2].item()),
                    ]
                )
                last_log_t = sim_time

            scene.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
            count += 1
            scene.update(sim_dt)

            if follow_cam is not None:
                follow_cam.step(uav.data.root_state_w)
    finally:
        if debug_interval is not None and last_debug_len > 0:
            print()
        if log_file is not None:
            log_file.close()
            print(f"[INFO]: CSV log saved: {args_cli.log_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-15.0, -8.0, 8.0], [0.0, 0.0, args_cli.target_alt])

    scene_cfg = BasicFixedWing1SceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    uav = scene["uav"]
    _init_airborne_state(
        uav=uav,
        scene=scene,
        altitude=float(args_cli.start_alt),
        speed=float(args_cli.start_speed),
        yaw_deg=float(args_cli.start_yaw_deg),
    )
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    print("[INFO]: Airborne init complete. Running circle debug.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
