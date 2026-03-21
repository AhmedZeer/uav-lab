import argparse
import csv
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-wing scene using per-surface aerodynamic modeling.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--mode",
    type=str,
    default="minimal_controller",
    choices=["open_loop", "minimal_controller"],
    help="Control mode.",
)
parser.add_argument("--roll", type=float, default=0.0, help="Open-loop roll command [-1, 1].")
parser.add_argument("--pitch", type=float, default=0.0, help="Open-loop pitch command [-1, 1].")
parser.add_argument("--yaw", type=float, default=0.0, help="Open-loop yaw command [-1, 1].")
parser.add_argument("--thrust", type=float, default=0.58, help="Open-loop thrust command [0, 1].")

parser.add_argument("--radius", type=float, default=80.0, help="Target circle radius [m].")
parser.add_argument("--center_x", type=float, default=0.0, help="Circle center x [m].")
parser.add_argument("--center_y", type=float, default=0.0, help="Circle center y [m].")
parser.add_argument("--clockwise", action="store_true", help="Use clockwise circle direction.")
parser.add_argument("--target_speed", type=float, default=20.0, help="Target speed [m/s].")
parser.add_argument("--target_alt", type=float, default=20.0, help="Target altitude [m].")
parser.add_argument("--max_thrust", type=float, default=100.0, help="Maximum thrust [N].")
parser.add_argument("--thrust_tau", type=float, default=0.02, help="First-order thrust lag time constant [s].")

parser.add_argument("--k_radial", type=float, default=0.05, help="Radial error gain.")
parser.add_argument("--k_heading", type=float, default=1.2, help="Heading-to-roll gain.")
parser.add_argument("--k_roll", type=float, default=2.0, help="Roll angle P gain.")
parser.add_argument("--k_p", type=float, default=0.35, help="Roll-rate damping gain.")
parser.add_argument("--k_alt", type=float, default=0.03, help="Altitude-to-pitch gain.")
parser.add_argument("--k_vz", type=float, default=0.08, help="Vertical speed damping gain.")
parser.add_argument("--k_pitch", type=float, default=1.8, help="Pitch angle P gain.")
parser.add_argument("--k_q", type=float, default=0.3, help="Pitch-rate damping gain.")
parser.add_argument("--k_speed", type=float, default=0.03, help="Speed-to-thrust gain.")
parser.add_argument("--thrust_trim", type=float, default=0.58, help="Throttle trim [0..1].")
parser.add_argument("--k_beta", type=float, default=0.4, help="Sideslip-to-yaw damping gain.")
parser.add_argument("--k_r", type=float, default=0.15, help="Yaw-rate damping gain.")
parser.add_argument("--max_bank_deg", type=float, default=45.0, help="Max commanded bank angle [deg].")
parser.add_argument("--max_pitch_deg", type=float, default=18.0, help="Max commanded pitch angle [deg].")

parser.add_argument("--start_alt", type=float, default=20.0, help="Initial altitude [m].")
parser.add_argument("--start_speed", type=float, default=18.0, help="Initial forward speed [m/s].")
parser.add_argument("--start_yaw_deg", type=float, default=0.0, help="Initial yaw [deg].")
parser.add_argument("--debug_print_hz", type=float, default=5.0, help="Console print rate [Hz].")
parser.add_argument(
    "--surface_reference_mode",
    type=str,
    default="config_minus_com",
    choices=["config", "config_minus_com", "zero"],
    help="Reference vector used for omega x r and r x F in the surface model.",
)

parser.add_argument("--log_csv", action="store_true", help="Write telemetry CSV.")
parser.add_argument("--log_path", type=str, default="logs/surfaces_debug.csv", help="Telemetry CSV path.")
parser.add_argument("--log_hz", type=float, default=50.0, help="CSV logging rate [Hz]. Set <= 0 for every step.")
parser.add_argument("--log_env_id", type=int, default=0, help="Environment index for logging.")

parser.add_argument("--follow_cam", action="store_true", help="Enable smoothed chase camera.")
parser.add_argument("--follow_cam_env_id", type=int, default=0, help="Environment index for chase camera.")
parser.add_argument("--follow_cam_distance", type=float, default=10.0, help="Chase distance behind UAV [m].")
parser.add_argument("--follow_cam_height", type=float, default=3.0, help="Chase camera height above UAV [m].")
parser.add_argument("--follow_cam_lookahead", type=float, default=6.0, help="Look-ahead distance in front of UAV [m].")
parser.add_argument("--follow_cam_smooth_tau", type=float, default=0.35, help="Smoothing time constant [s].")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils
from uav_lab_1.robots.follow_camera import FollowCameraConfig, SmoothedFollowCamera
from uav_lab_1.robots.surface_aero import SurfaceAeroModel, default_fixedwing_surface_configs
from uav_lab_1.scenes import BasicFixedWing1SceneCfg

MAX_FORCE = 2000.0
MAX_TORQUE = 2000.0


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


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
    root_vel[:, 1] = speed * torch.sin(yaw)
    root_vel[:, 2] = 0.0
    root_vel[:, 3:6] = 0.0

    uav.write_root_pose_to_sim(root_pose, env_ids)
    uav.write_root_velocity_to_sim(root_vel, env_ids)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    uav = scene["uav"]
    body_ids, _ = uav.find_bodies("body")
    if len(body_ids) == 0:
        body_ids = [0]
    body_id = int(body_ids[0])

    device = uav.device
    num_envs = uav.num_instances
    body_ids_t = torch.tensor([body_id], device=device, dtype=torch.int32)
    env_i = int(max(0, min(num_envs - 1, int(args_cli.log_env_id)))) if num_envs > 0 else 0

    debug_interval = (1.0 / float(args_cli.debug_print_hz)) if float(args_cli.debug_print_hz) > 0.0 else None
    last_debug_t = -1.0e9
    last_debug_len = 0

    surface_cfgs = default_fixedwing_surface_configs()
    surface_model = SurfaceAeroModel(surface_cfgs, num_envs=num_envs, sim_dt=sim_dt, device=device)
    surface_names = surface_model.surface_names

    # Surface command mix from [roll, pitch, yaw] -> [left, right, h_tail, v_tail, main]
    surface_mix = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # left aileron
            [-1.0, 0.0, 0.0],  # right aileron
            [0.0, 1.0, 0.0],   # horizontal tail
            [0.0, 0.0, -1.0],  # vertical tail
            [0.0, -1.0, 0.0],  # main wing flap assist
        ],
        device=device,
        dtype=torch.float32,
    )

    throttle_state = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    thrust_tau = max(1.0e-4, float(args_cli.thrust_tau))

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
        header = [
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
            "throttle_state",
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
        for name in surface_names:
            header += [f"alpha_{name}", f"act_{name}"]
        log_writer.writerow(header)
        print(f"[INFO]: Logging to {args_cli.log_path}")

    center = torch.tensor([float(args_cli.center_x), float(args_cli.center_y)], device=device)
    turn_sign = -1.0 if args_cli.clockwise else 1.0
    max_bank = math.radians(float(args_cli.max_bank_deg))
    max_pitch = math.radians(float(args_cli.max_pitch_deg))
    g = 9.81

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

    print(
        f"[INFO]: Running mode={args_cli.mode} with surface aero "
        f"(num_surfaces={surface_model.num_surfaces}, max_thrust={args_cli.max_thrust:.1f}N, "
        f"ref_mode={args_cli.surface_reference_mode})."
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

            if args_cli.mode == "open_loop":
                roll_cmd = torch.full((num_envs,), float(args_cli.roll), device=device)
                pitch_cmd = torch.full((num_envs,), float(args_cli.pitch), device=device)
                yaw_cmd = torch.full((num_envs,), float(args_cli.yaw), device=device)
                thrust_cmd = torch.full((num_envs,), float(args_cli.thrust), device=device)
                radius_err = torch.zeros((num_envs,), device=device)
                heading_err = torch.zeros((num_envs,), device=device)
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
                roll_cmd = _clamp(float(args_cli.k_roll) * roll_err - float(args_cli.k_p) * w_body[:, 0], -1.0, 1.0)

                alt_err = float(args_cli.target_alt) - pos_world[:, 2]
                pitch_des = float(args_cli.k_alt) * alt_err - float(args_cli.k_vz) * v_world[:, 2]
                pitch_des = torch.clamp(pitch_des, -max_pitch, max_pitch)
                pitch_err = _wrap_pi(pitch_des - pitch)
                pitch_cmd = _clamp(float(args_cli.k_pitch) * pitch_err - float(args_cli.k_q) * w_body[:, 1], -1.0, 1.0)

                speed_err = float(args_cli.target_speed) - v_body[:, 0]
                thrust_cmd = _clamp(float(args_cli.thrust_trim) + float(args_cli.k_speed) * speed_err, 0.0, 1.0)

                safe_speed = torch.clamp(torch.linalg.norm(v_body, dim=-1), min=0.1)
                beta = torch.asin(torch.clamp(-v_body[:, 1] / safe_speed, min=-1.0, max=1.0))
                yaw_cmd = _clamp(-float(args_cli.k_beta) * beta - float(args_cli.k_r) * w_body[:, 2], -1.0, 1.0)

            thrust_cmd = _clamp(thrust_cmd, 0.0, 1.0)
            throttle_state = throttle_state + (sim_dt / thrust_tau) * (thrust_cmd - throttle_state)
            throttle_state = _clamp(throttle_state, 0.0, 1.0)
            thrust_force = torch.stack(
                [
                    throttle_state * float(args_cli.max_thrust),
                    torch.zeros_like(throttle_state),
                    torch.zeros_like(throttle_state),
                ],
                dim=-1,
            )

            cmd_vec = torch.stack([roll_cmd, pitch_cmd, yaw_cmd], dim=-1)  # (N, 3)
            surface_cmd = _clamp(torch.matmul(cmd_vec, surface_mix.T), -1.0, 1.0)  # (N, S)
            body_com_b = uav.data.body_com_pos_b[:, body_id]
            aero_out = surface_model.step(
                v_body=v_body,
                w_body=w_body,
                cmd=surface_cmd,
                body_com_b=body_com_b,
                reference_mode=args_cli.surface_reference_mode,
            )

            total_forces = _clamp(thrust_force + aero_out["force_b"], -MAX_FORCE, MAX_FORCE)
            total_torques = _clamp(aero_out["torque_b"], -MAX_TORQUE, MAX_TORQUE)

            com_positions = torch.zeros((num_envs, 1, 3), device=device)
            com_positions[:, 0, :] = body_com_b

            forces = torch.zeros((num_envs, 1, 3), device=device)
            forces[:, 0, :] = total_forces

            torques = torch.zeros((num_envs, 1, 3), device=device)
            torques[:, 0, :] = total_torques

            composer = uav.instantaneous_wrench_composer
            composer.add_forces_and_torques(forces=forces, positions=com_positions, body_ids=body_ids_t)
            composer.add_forces_and_torques(torques=torques, body_ids=body_ids_t)

            if debug_interval is not None and (sim_time - last_debug_t) >= debug_interval:
                line = (
                    f"env{env_i} mode={args_cli.mode} "
                    f"v=({v_body[env_i,0].item():6.2f},{v_body[env_i,1].item():6.2f},{v_body[env_i,2].item():6.2f}) "
                    f"pqr=({w_body[env_i,0].item():6.2f},{w_body[env_i,1].item():6.2f},{w_body[env_i,2].item():6.2f}) "
                    f"cmd=({roll_cmd[env_i].item():5.2f},{pitch_cmd[env_i].item():5.2f},{yaw_cmd[env_i].item():5.2f},"
                    f"{thrust_cmd[env_i].item():5.2f}) "
                    f"F=({total_forces[env_i,0].item():7.2f},{total_forces[env_i,1].item():7.2f},{total_forces[env_i,2].item():7.2f}) "
                    f"T=({total_torques[env_i,0].item():7.2f},{total_torques[env_i,1].item():7.2f},{total_torques[env_i,2].item():7.2f})"
                )
                pad = " " * max(0, last_debug_len - len(line))
                print(f"\r{line}{pad}", end="", flush=True)
                last_debug_t = sim_time
                last_debug_len = len(line)

            if log_writer is not None and (log_interval == 0.0 or (sim_time - last_log_t) >= log_interval):
                row = [
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
                    float(radius_err[env_i].item()),
                    float(heading_err[env_i].item()),
                    float(thrust_cmd[env_i].item()),
                    float(throttle_state[env_i].item()),
                    float(roll_cmd[env_i].item()),
                    float(pitch_cmd[env_i].item()),
                    float(yaw_cmd[env_i].item()),
                    float(total_forces[env_i, 0].item()),
                    float(total_forces[env_i, 1].item()),
                    float(total_forces[env_i, 2].item()),
                    float(total_torques[env_i, 0].item()),
                    float(total_torques[env_i, 1].item()),
                    float(total_torques[env_i, 2].item()),
                ]
                for s in range(surface_model.num_surfaces):
                    row.append(float(aero_out["alpha"][env_i, s].item()))
                    row.append(float(aero_out["actuation"][env_i, s].item()))
                log_writer.writerow(row)
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

    print("[INFO]: Airborne init complete. Running surface-aero scene.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
