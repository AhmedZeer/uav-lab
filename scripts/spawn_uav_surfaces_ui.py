import argparse
import csv
import math
import os
from dataclasses import replace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manual fixed-wing scene using per-surface aerodynamic modeling.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--ui", action="store_true", help="Enable UI controls for thrust/roll/pitch/yaw.")
parser.add_argument("--vis_markers", action="store_true", help="Show frame/force visualization markers.")
parser.add_argument("--force_vis_scale", type=float, default=0.02, help="Scale for force arrow length.")
parser.add_argument("--thrust", type=float, default=0.0, help="Initial thrust command [0, 1].")
parser.add_argument("--roll", type=float, default=0.0, help="Initial roll command [-1, 1].")
parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch command [-1, 1].")
parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw command [-1, 1].")
parser.add_argument("--max_thrust", type=float, default=200.0, help="Maximum thrust [N].")
parser.add_argument("--thrust_tau", type=float, default=0.02, help="First-order thrust lag time constant [s].")
parser.add_argument(
    "--propeller_anim",
    action="store_true",
    help="Enable propeller joint actuation for visual spin. This can affect dynamics.",
)
parser.add_argument(
    "--propeller_anim_effort",
    type=float,
    default=25.0,
    help="Propeller joint effort target at full throttle for visual animation.",
)
parser.add_argument(
    "--propeller_anim_threshold",
    type=float,
    default=0.1,
    help="Throttle threshold above which propeller animation effort is applied.",
)
parser.add_argument(
    "--propeller_anim_sign",
    type=float,
    default=1.0,
    help="Sign applied to the propeller joint effort. Flip to -1 if the mesh spins the wrong way.",
)
parser.add_argument("--start_alt", type=float, default=10.0, help="Initial altitude [m].")
parser.add_argument("--start_speed", type=float, default=40.0, help="Initial forward speed [m/s].")
parser.add_argument("--start_yaw_deg", type=float, default=0.0, help="Initial yaw [deg].")
parser.add_argument("--debug_print_hz", type=float, default=5.0, help="Console print rate [Hz].")
parser.add_argument("--telemetry_ui", action="store_true", help="Show live aerodynamic telemetry in an Isaac Sim window.")
parser.add_argument("--telemetry_hz", type=float, default=10.0, help="Telemetry UI update rate [Hz]. Set <= 0 for every step.")
parser.add_argument(
    "--log_frame_debug",
    action="store_true",
    help="Log root/root-link/body-link frame state side by side when available.",
)
parser.add_argument(
    "--log_surface_debug",
    action="store_true",
    help="Log per-surface local velocity, reference arm, and split force/torque terms.",
)

parser.add_argument("--log_csv", action="store_true", help="Write telemetry CSV.")
parser.add_argument("--log_path", type=str, default="logs/surfaces_manual_debug.csv", help="Telemetry CSV path.")
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from uav_lab_1.robots.follow_camera import FollowCameraConfig, SmoothedFollowCamera
from uav_lab_1.robots.surface_aero import SurfaceAeroModel, default_fixedwing_surface_configs
from uav_lab_1.scenes import BasicFixedWing1SceneCfg, MudFixedWing1SceneCfg

MAX_FORCE = 10000.0
MAX_TORQUE = 10000.0


class ManualSurfaceControlWindow:
    def __init__(self, thrust=0.0, roll=0.0, pitch=0.0, yaw=0.0, max_thrust=100.0):
        import omni.ui as ui

        self._window = ui.Window("Surface Aero Manual Controls", width=360, height=260)
        self._models = {
            "thrust": ui.SimpleFloatModel(thrust),
            "roll": ui.SimpleFloatModel(roll),
            "pitch": ui.SimpleFloatModel(pitch),
            "yaw": ui.SimpleFloatModel(yaw),
            "max_thrust": ui.SimpleFloatModel(max_thrust),
        }
        self._build_ui(ui)

    def _build_ui(self, ui):
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Propulsion", height=20)
                self._create_drag(ui, "Thrust [0..1]", self._models["thrust"], 0.0, 1.0, 0.01)
                self._create_drag(ui, "Max Thrust [N]", self._models["max_thrust"], 0.0, 500.0, 1.0)
                ui.Separator(height=2)
                ui.Label("Surface Commands", height=20)
                self._create_drag(ui, "Roll [-1..1]", self._models["roll"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Pitch [-1..1]", self._models["pitch"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Yaw [-1..1]", self._models["yaw"], -1.0, 1.0, 0.01)
                ui.Spacer(height=6)
                ui.Button("Zero All", clicked_fn=self.zero_all)

    def _create_drag(self, ui, label, model, min_val, max_val, step):
        with ui.HStack(height=24):
            ui.Label(label, width=120)
            ui.FloatDrag(model, min=min_val, max=max_val, step=step)

    def zero_all(self):
        self._models["thrust"].as_float = 0.0
        self._models["roll"].as_float = 0.0
        self._models["pitch"].as_float = 0.0
        self._models["yaw"].as_float = 0.0

    def get_inputs(self):
        return (
            float(self._models["thrust"].as_float),
            float(self._models["roll"].as_float),
            float(self._models["pitch"].as_float),
            float(self._models["yaw"].as_float),
            float(self._models["max_thrust"].as_float),
        )


class SurfaceTelemetryWindow:
    def __init__(self, surface_names: list[str], env_id: int):
        import omni.ui as ui

        self._surface_names = surface_names
        self._window = ui.Window("Surface Aero Telemetry", width=560, height=420)
        self._header = None
        self._summary = None
        self._surface_labels = []

        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                self._header = ui.Label(f"env {env_id}", height=20)
                self._summary = ui.Label("", height=120, alignment=ui.Alignment.LEFT_TOP)
                ui.Separator(height=2)
                for name in surface_names:
                    self._surface_labels.append(ui.Label(f"{name}: --", height=26, alignment=ui.Alignment.LEFT_TOP))

    def update(
        self,
        env_id: int,
        speed: float,
        body_rates: tuple[float, float, float],
        total_force: tuple[float, float, float],
        total_torque: tuple[float, float, float],
        surface_actuation: list[float],
        surface_torque: list[tuple[float, float, float]],
    ):
        force_amp = math.sqrt(sum(component * component for component in total_force))
        torque_amp = math.sqrt(sum(component * component for component in total_torque))
        p_rate, q_rate, r_rate = body_rates

        if self._header is not None:
            self._header.text = f"env {env_id}"
        if self._summary is not None:
            self._summary.text = (
                f"Speed: {speed:7.2f} m/s\n"
                f"Force amplitude: {force_amp:7.2f} N\n"
                f"Torque amplitude: {torque_amp:7.2f} N m\n"
                f"Body rates [p q r]: {p_rate:7.3f}  {q_rate:7.3f}  {r_rate:7.3f} rad/s\n"
                f"Total force [x y z]: {total_force[0]:7.2f}  {total_force[1]:7.2f}  {total_force[2]:7.2f} N\n"
                f"Total torque [x y z]: {total_torque[0]:7.2f}  {total_torque[1]:7.2f}  {total_torque[2]:7.2f} N m"
            )

        for label, name, actuation, torque in zip(
            self._surface_labels, self._surface_names, surface_actuation, surface_torque, strict=False
        ):
            torque_mag = math.sqrt(sum(component * component for component in torque))
            label.text = (
                f"{name}: act={actuation:+5.2f}  |T|={torque_mag:7.3f} N m  "
                f"T=[{torque[0]:+7.3f}, {torque[1]:+7.3f}, {torque[2]:+7.3f}]"
            )


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def _quat_from_two_vectors(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    v0_n = math_utils.normalize(v0)
    v1_n = math_utils.normalize(v1)
    dot = torch.sum(v0_n * v1_n, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.cross(v0_n, v1_n, dim=-1)
    w = 1.0 + dot
    quat = torch.cat([w, cross], dim=-1)

    near_opposite = dot.squeeze(-1) < -0.9999
    if torch.any(near_opposite):
        axis = torch.zeros_like(v0_n)
        axis[:, 0] = 1.0
        axis = torch.cross(v0_n, axis, dim=-1)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        alt_axis = torch.zeros_like(v0_n)
        alt_axis[:, 1] = 1.0
        axis = torch.where(axis_norm > 1e-6, axis, torch.cross(v0_n, alt_axis, dim=-1))
        axis = math_utils.normalize(axis)
        quat_alt = torch.cat([torch.zeros_like(dot), axis], dim=-1)
        quat = torch.where(near_opposite.unsqueeze(-1), quat_alt, quat)
    return math_utils.normalize(quat)


def _create_visual_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/uavSurfaceMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.4, 0.4, 0.4),
            ),
            "arrow_x": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.2, 0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


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


def _vec3_or_nan(x: torch.Tensor | None, env_idx: int) -> list[float]:
    if x is None:
        return [float("nan"), float("nan"), float("nan")]
    v = x[env_idx]
    return [float(v[0].item()), float(v[1].item()), float(v[2].item())]


def _quat_or_nan(x: torch.Tensor | None, env_idx: int) -> list[float]:
    if x is None:
        return [float("nan"), float("nan"), float("nan"), float("nan")]
    q = x[env_idx]
    return [float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item())]


def _indexed_tensor(x: torch.Tensor | None, index: int, width: int) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim == 2 and x.shape[-1] >= width:
        return x[:, :width]
    if x.ndim == 3 and index < x.shape[1] and x.shape[-1] >= width:
        return x[:, index, :width]
    return None


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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, markers: VisualizationMarkers | None):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    uav = scene["uav"]
    body_ids, _ = uav.find_bodies("body")
    if len(body_ids) == 0:
        body_ids = [0]
    body_id = int(body_ids[0])
    # propeller_joint_ids, _ = uav.find_joints("propeller_joint")
    propeller_joint_ids, _ = uav.find_joints("propeller_joint")
    propeller_joint_ids = [int(j) for j in propeller_joint_ids]

    device = uav.device
    num_envs = uav.num_instances
    env_i = int(max(0, min(num_envs - 1, int(args_cli.log_env_id)))) if num_envs > 0 else 0
    body_ids_t = torch.tensor([body_id], device=device, dtype=torch.int32)

    thrust_cmd = float(args_cli.thrust)
    roll_cmd = float(args_cli.roll)
    pitch_cmd = float(args_cli.pitch)
    yaw_cmd = float(args_cli.yaw)
    max_thrust = float(args_cli.max_thrust)

    debug_interval = (1.0 / float(args_cli.debug_print_hz)) if float(args_cli.debug_print_hz) > 0.0 else None
    last_debug_t = -1.0e9
    last_debug_len = 0

    control_ui = None
    if args_cli.ui:
        try:
            control_ui = ManualSurfaceControlWindow(
                thrust=thrust_cmd,
                roll=roll_cmd,
                pitch=pitch_cmd,
                yaw=yaw_cmd,
                max_thrust=max_thrust,
            )
            print(f"[INFO]: trpy = {thrust_cmd}, {roll_cmd}, {pitch_cmd}, {yaw_cmd}")
        except Exception as e:
            print(f"[WARN]: Failed to create UI controls: {e}")

    surface_cfgs = default_fixedwing_surface_configs()
    surface_model = SurfaceAeroModel(surface_cfgs, num_envs=num_envs, sim_dt=sim_dt, device=device)
    surface_names = surface_model.surface_names
    telemetry_ui = None
    telemetry_interval = (1.0 / float(args_cli.telemetry_hz)) if float(args_cli.telemetry_hz) > 0.0 else 0.0
    last_telemetry_t = -1.0e9
    if args_cli.telemetry_ui:
        try:
            telemetry_ui = SurfaceTelemetryWindow(surface_names=surface_names, env_id=env_i)
            debug_interval = None
        except Exception as e:
            print(f"[WARN]: Failed to create telemetry UI: {e}")

    surface_mix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    throttle_state = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    thrust_tau = max(1.0e-4, float(args_cli.thrust_tau))
    propeller_effort_target = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)
    propeller_joint_speed = torch.zeros((num_envs,), device=device, dtype=torch.float32)

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
            "thrust_cmd",
            "throttle_state",
            "roll_cmd",
            "pitch_cmd",
            "yaw_cmd",
            "max_thrust_n",
            "force_x",
            "force_y",
            "force_z",
            "torque_x",
            "torque_y",
            "torque_z",
        ]
        if args_cli.log_frame_debug:
            header += [
                "root_link_pos_w_x",
                "root_link_pos_w_y",
                "root_link_pos_w_z",
                "root_link_quat_w",
                "root_link_quat_x",
                "root_link_quat_y",
                "root_link_quat_z",
                "body_link_pos_w_x",
                "body_link_pos_w_y",
                "body_link_pos_w_z",
                "body_link_quat_w",
                "body_link_quat_x",
                "body_link_quat_y",
                "body_link_quat_z",
                "body_link_lin_vel_w_x",
                "body_link_lin_vel_w_y",
                "body_link_lin_vel_w_z",
                "body_link_ang_vel_w_x",
                "body_link_ang_vel_w_y",
                "body_link_ang_vel_w_z",
                "root_to_root_link_dx",
                "root_to_root_link_dy",
                "root_to_root_link_dz",
                "root_to_body_link_dx",
                "root_to_body_link_dy",
                "root_to_body_link_dz",
                "body_com_b_x",
                "body_com_b_y",
                "body_com_b_z",
            ]
        for name in surface_names:
            header += [f"alpha_{name}", f"act_{name}"]
        if args_cli.log_surface_debug:
            for name in surface_names:
                header += [
                    f"cfg_pos_{name}_x",
                    f"cfg_pos_{name}_y",
                    f"cfg_pos_{name}_z",
                    f"ref_pos_{name}_x",
                    f"ref_pos_{name}_y",
                    f"ref_pos_{name}_z",
                    f"vlocal_{name}_x",
                    f"vlocal_{name}_y",
                    f"vlocal_{name}_z",
                    f"surf_force_{name}_x",
                    f"surf_force_{name}_y",
                    f"surf_force_{name}_z",
                    f"surf_total_torque_{name}_x",
                    f"surf_total_torque_{name}_y",
                    f"surf_total_torque_{name}_z",
                    f"surf_aero_torque_{name}_x",
                    f"surf_aero_torque_{name}_y",
                    f"surf_aero_torque_{name}_z",
                    f"surf_force_torque_{name}_x",
                    f"surf_force_torque_{name}_y",
                    f"surf_force_torque_{name}_z",
                    f"fwd_speed_{name}",
                    f"lift_speed_{name}",
                ]
        log_writer.writerow(header)
        print(f"[INFO]: Logging to {args_cli.log_path}")

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
        "[INFO]: Running manual surface-aero scene "
        f"(num_surfaces={surface_model.num_surfaces}, ui={args_cli.ui}, max_thrust={max_thrust:.1f}N, "
        f"(cli_trpy={args_cli.thrust}, {args_cli.roll}, {args_cli.pitch}, {args_cli.yaw}"
    )
    if args_cli.propeller_anim and propeller_joint_ids:
        print(
            "[INFO]: Propeller effort animation enabled "
            f"(joint_ids={propeller_joint_ids}, max_effort={args_cli.propeller_anim_effort:.1f}, "
            f"threshold={args_cli.propeller_anim_threshold:.2f}, sign={args_cli.propeller_anim_sign:+.1f})."
        )
    elif propeller_joint_ids:
        print("[INFO]: Propeller animation is disabled. Use --propeller_anim to enable joint actuation.")
    else:
        print("[WARN]: Could not find 'propeller_joint'; propeller animation is disabled.")

    try:
        while simulation_app.is_running():
            if control_ui is not None:
                thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd, max_thrust = control_ui.get_inputs()

            state = uav.data.root_state_w
            quat_wxyz = state[:, 3:7]
            pos_world = state[:, 0:3]
            v_world = state[:, 7:10]
            w_world = state[:, 10:13]
            root_link_pose_w = _indexed_tensor(getattr(uav.data, "root_link_pose_w", None), 0, 7)
            body_link_pose_w = _indexed_tensor(getattr(uav.data, "body_link_pose_w", None), body_id, 7)
            body_state_w = _indexed_tensor(getattr(uav.data, "body_state_w", None), body_id, 13)
            body_link_lin_vel_w = body_state_w[:, 7:10] if body_state_w is not None else None
            body_link_ang_vel_w = body_state_w[:, 10:13] if body_state_w is not None else None
            body_com_b = uav.data.body_com_pos_b[:, body_id]

            v_body = math_utils.quat_apply_inverse(quat_wxyz, v_world)
            w_body = math_utils.quat_apply_inverse(quat_wxyz, w_world)
            roll, pitch, yaw = _euler_from_quat_wxyz(quat_wxyz)

            thrust_cmd_t = _clamp(torch.full((num_envs,), thrust_cmd, device=device), 0.0, 1.0)
            throttle_state = throttle_state + (sim_dt / thrust_tau) * (thrust_cmd_t - throttle_state)
            throttle_state = _clamp(throttle_state, 0.0, 1.0)
            if args_cli.propeller_anim and propeller_joint_ids:
                propeller_effort_target = torch.where(
                    throttle_state.unsqueeze(-1) >= float(args_cli.propeller_anim_threshold),
                    throttle_state.unsqueeze(-1)
                    * float(args_cli.propeller_anim_effort)
                    * float(args_cli.propeller_anim_sign),
                    torch.zeros_like(propeller_effort_target),
                )
                uav.set_joint_effort_target(propeller_effort_target, joint_ids=propeller_joint_ids)
                propeller_joint_speed = uav.data.joint_vel[:, propeller_joint_ids[0]]
            else:
                propeller_effort_target.zero_()
                propeller_joint_speed.zero_()
            thrust_force = torch.stack(
                [
                    throttle_state * float(max_thrust),
                    torch.zeros_like(throttle_state),
                    torch.zeros_like(throttle_state),
                ],
                dim=-1,
            )

            cmd_vec = torch.stack(
                [
                    torch.full((num_envs,), roll_cmd, device=device),
                    torch.full((num_envs,), pitch_cmd, device=device),
                    torch.full((num_envs,), yaw_cmd, device=device),
                ],
                dim=-1,
            )
            cmd_vec = _clamp(cmd_vec, -1.0, 1.0)
            surface_cmd = _clamp(torch.matmul(cmd_vec, surface_mix.T), -1.0, 1.0)
            aero_out = surface_model.step(
                v_body=v_body,
                w_body=w_body,
                cmd=surface_cmd,
                body_com_b=body_com_b,
                reference_mode="config",
            )

            total_forces = _clamp(thrust_force + aero_out["force_b"], -MAX_FORCE, MAX_FORCE) #* 0.5
            total_torques = _clamp(aero_out["torque_b"], -MAX_TORQUE, MAX_TORQUE) #* 0.3

            if telemetry_ui is not None and (telemetry_interval == 0.0 or (sim_time - last_telemetry_t) >= telemetry_interval):
                telemetry_ui.update(
                    env_id=env_i,
                    speed=float(torch.linalg.norm(v_body[env_i]).item()),
                    body_rates=(
                        float(w_body[env_i, 0].item()),
                        float(w_body[env_i, 1].item()),
                        float(w_body[env_i, 2].item()),
                    ),
                    total_force=(
                        float(total_forces[env_i, 0].item()),
                        float(total_forces[env_i, 1].item()),
                        float(total_forces[env_i, 2].item()),
                    ),
                    total_torque=(
                        float(total_torques[env_i, 0].item()),
                        float(total_torques[env_i, 1].item()),
                        float(total_torques[env_i, 2].item()),
                    ),
                    surface_actuation=[float(aero_out["actuation"][env_i, s].item()) for s in range(surface_model.num_surfaces)],
                    surface_torque=[
                        (
                            float(aero_out["surface_torque_b"][env_i, s, 0].item()),
                            float(aero_out["surface_torque_b"][env_i, s, 1].item()),
                            float(aero_out["surface_torque_b"][env_i, s, 2].item()),
                        )
                        for s in range(surface_model.num_surfaces)
                    ],
                )
                last_telemetry_t = sim_time

            com_positions = torch.zeros((num_envs, 1, 3), device=device)
            com_positions[:, 0, :] = body_com_b

            forces = torch.zeros((num_envs, 1, 3), device=device)
            forces[:, 0, :] = total_forces

            torques = torch.zeros((num_envs, 1, 3), device=device)
            torques[:, 0, :] = total_torques

            composer = uav.instantaneous_wrench_composer
            composer.add_forces_and_torques(forces=forces, positions=com_positions, body_ids=body_ids_t)
            composer.add_forces_and_torques(torques=torques, body_ids=body_ids_t)

            if markers is not None:
                pos = uav.data.root_link_pose_w[:, 0:3]
                quat = uav.data.root_link_pose_w[:, 3:7]
                pos_vis = pos + torch.tensor([0.0, 0.0, 0.4], device=device)
                force_world = math_utils.quat_apply(quat, total_forces)
                mag = torch.linalg.norm(force_world, dim=-1, keepdim=True)
                fallback_dir = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(force_world)
                dir_world = torch.where(mag > 1e-6, force_world / mag, fallback_dir)
                base = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(dir_world)
                arrow_quat = _quat_from_two_vectors(base, dir_world)

                translations = torch.cat([pos_vis, pos_vis], dim=0)
                orientations = torch.cat([quat, arrow_quat], dim=0)
                frame_scale = torch.full((num_envs, 3), 0.4, device=device)
                arrow_scale_x = torch.clamp(mag * float(args_cli.force_vis_scale), min=0.05, max=5.0)
                arrow_scale = torch.cat([arrow_scale_x, torch.full((num_envs, 2), 0.2, device=device)], dim=-1)
                scales = torch.cat([frame_scale, arrow_scale], dim=0)
                marker_indices = torch.cat(
                    [
                        torch.zeros(num_envs, dtype=torch.int64, device=device),
                        torch.ones(num_envs, dtype=torch.int64, device=device),
                    ],
                    dim=0,
                )
                markers.visualize(
                    translations=translations,
                    orientations=orientations,
                    scales=scales,
                    marker_indices=marker_indices,
                )

            if debug_interval is not None and (sim_time - last_debug_t) >= debug_interval:
                root_link_delta = [float("nan"), float("nan"), float("nan")]
                if root_link_pose_w is not None:
                    root_link_delta = _vec3_or_nan(root_link_pose_w[:, 0:3] - pos_world, env_i)
                line = (
                    f"env{env_i} v=({v_body[env_i,0].item():6.2f},{v_body[env_i,1].item():6.2f},{v_body[env_i,2].item():6.2f}) "
                    f"AeroT=({aero_out['torque_b'][env_i,0].item():7.2f},{aero_out['torque_b'][env_i,1].item():7.2f},{aero_out['torque_b'][env_i,2].item():7.2f}) "
                    f"AeroF=({aero_out['force_b'][env_i,0].item():7.2f},{aero_out['force_b'][env_i,1].item():7.2f},{aero_out['force_b'][env_i,2].item():7.2f}) "
                    f"pqr=({w_body[env_i,0].item():6.2f},{w_body[env_i,1].item():6.2f},{w_body[env_i,2].item():6.2f}) "
                    f"com=({body_com_b[env_i,0].item():6.3f},{body_com_b[env_i,1].item():6.3f},{body_com_b[env_i,2].item():6.3f}) "
                    f"drl=({root_link_delta[0]:6.3f},{root_link_delta[1]:6.3f},{root_link_delta[2]:6.3f}) "
                    f"cmd=({roll_cmd:5.2f},{pitch_cmd:5.2f},{yaw_cmd:5.2f},{thrust_cmd:5.2f}) "
                    f"F=({total_forces[env_i,0].item():7.2f},{total_forces[env_i,1].item():7.2f},{total_forces[env_i,2].item():7.2f}) "
                    f"T=({total_torques[env_i,0].item():7.2f},{total_torques[env_i,1].item():7.2f},{total_torques[env_i,2].item():7.2f})\n"
                    f"PropEff=({propeller_effort_target[env_i].item():7.2f}) "
                    f"PropVel=({propeller_joint_speed[env_i].item():7.2f})"
                )
                pad = " " * max(0, last_debug_len - len(line))
                print(f"\r{line}{pad}", end="", flush=True)
                last_debug_t = sim_time
                last_debug_len = len(line)

            if log_writer is not None and (log_interval == 0.0 or (sim_time - last_log_t) >= log_interval):
                body_link_pose_for_log = body_link_pose_w
                if body_link_pose_for_log is None and body_state_w is not None:
                    body_link_pose_for_log = body_state_w[:, :7]
                row = [
                    float(sim_time),
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
                    float(thrust_cmd),
                    float(throttle_state[env_i].item()),
                    float(roll_cmd),
                    float(pitch_cmd),
                    float(yaw_cmd),
                    float(max_thrust),
                    float(total_forces[env_i, 0].item()),
                    float(total_forces[env_i, 1].item()),
                    float(total_forces[env_i, 2].item()),
                    float(total_torques[env_i, 0].item()),
                    float(total_torques[env_i, 1].item()),
                    float(total_torques[env_i, 2].item()),
                ]
                if args_cli.log_frame_debug:
                    root_link_delta = (
                        root_link_pose_w[:, 0:3] - pos_world if root_link_pose_w is not None else None
                    )
                    body_link_delta = (
                        body_link_pose_for_log[:, 0:3] - pos_world if body_link_pose_for_log is not None else None
                    )
                    row += [
                        *_vec3_or_nan(root_link_pose_w[:, 0:3] if root_link_pose_w is not None else None, env_i),
                        *_quat_or_nan(root_link_pose_w[:, 3:7] if root_link_pose_w is not None else None, env_i),
                        *_vec3_or_nan(body_link_pose_for_log[:, 0:3] if body_link_pose_for_log is not None else None, env_i),
                        *_quat_or_nan(body_link_pose_for_log[:, 3:7] if body_link_pose_for_log is not None else None, env_i),
                        *_vec3_or_nan(body_link_lin_vel_w, env_i),
                        *_vec3_or_nan(body_link_ang_vel_w, env_i),
                        *_vec3_or_nan(root_link_delta, env_i),
                        *_vec3_or_nan(body_link_delta, env_i),
                        *_vec3_or_nan(body_com_b, env_i),
                    ]
                for s in range(surface_model.num_surfaces):
                    row.append(float(aero_out["alpha"][env_i, s].item()))
                    row.append(float(aero_out["actuation"][env_i, s].item()))
                if args_cli.log_surface_debug:
                    for s in range(surface_model.num_surfaces):
                        row += [
                            float(aero_out["surface_config_pos_b"][env_i, s, 0].item()),
                            float(aero_out["surface_config_pos_b"][env_i, s, 1].item()),
                            float(aero_out["surface_config_pos_b"][env_i, s, 2].item()),
                            float(aero_out["surface_reference_pos_b"][env_i, s, 0].item()),
                            float(aero_out["surface_reference_pos_b"][env_i, s, 1].item()),
                            float(aero_out["surface_reference_pos_b"][env_i, s, 2].item()),
                            float(aero_out["surface_local_velocity_b"][env_i, s, 0].item()),
                            float(aero_out["surface_local_velocity_b"][env_i, s, 1].item()),
                            float(aero_out["surface_local_velocity_b"][env_i, s, 2].item()),
                            float(aero_out["surface_force_b"][env_i, s, 0].item()),
                            float(aero_out["surface_force_b"][env_i, s, 1].item()),
                            float(aero_out["surface_force_b"][env_i, s, 2].item()),
                            float(aero_out["surface_torque_b"][env_i, s, 0].item()),
                            float(aero_out["surface_torque_b"][env_i, s, 1].item()),
                            float(aero_out["surface_torque_b"][env_i, s, 2].item()),
                            float(aero_out["surface_aero_torque_b"][env_i, s, 0].item()),
                            float(aero_out["surface_aero_torque_b"][env_i, s, 1].item()),
                            float(aero_out["surface_aero_torque_b"][env_i, s, 2].item()),
                            float(aero_out["surface_force_torque_b"][env_i, s, 0].item()),
                            float(aero_out["surface_force_torque_b"][env_i, s, 1].item()),
                            float(aero_out["surface_force_torque_b"][env_i, s, 2].item()),
                            float(aero_out["surface_forward_speed"][env_i, s].item()),
                            float(aero_out["surface_lift_speed"][env_i, s].item()),
                        ]
                log_writer.writerow(row)
                last_log_t = sim_time

            scene.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
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
    sim.set_camera_view([-15.0, -8.0, 8.0], [0.0, 0.0, args_cli.start_alt])

    scene_cfg = MudFixedWing1SceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    markers = _create_visual_markers() if args_cli.vis_markers else None

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

    print("[INFO]: Airborne init complete. Running manual surface-aero scene.")
    run_simulator(sim, scene, markers)


if __name__ == "__main__":
    main()
    simulation_app.close()
