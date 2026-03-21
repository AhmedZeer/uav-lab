
import argparse
import csv
import os
from collections import deque

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--thrust", type=float, default=0.0, help="Normalized thrust command [0, 1].")
parser.add_argument("--roll", type=float, default=0.0, help="Roll command [-1, 1].")
parser.add_argument("--pitch", type=float, default=0.0, help="Pitch command [-1, 1].")
parser.add_argument("--yaw", type=float, default=0.0, help="Yaw command [-1, 1].")
parser.add_argument("--max_thrust", type=float, default=100.0, help="Maximum thrust in Newtons.")
parser.add_argument("--max_torque", type=float, default=50.0, help="Maximum body torque in N*m.")
parser.add_argument(
    "--debug_print_hz",
    type=float,
    default=10.0,
    help="In-place debug print rate (Hz). Set <= 0 to disable.",
)
parser.add_argument("--ui", action="store_true", help="Enable UI controls for thrust/roll/pitch/yaw.")
parser.add_argument("--vis_markers", action="store_true", help="Show frame/force visualization markers.")
parser.add_argument("--force_vis_scale", type=float, default=0.02, help="Scale for force arrow length.")
parser.add_argument("--log_csv", action="store_true", help="Write telemetry and inputs to CSV.")
parser.add_argument("--log_path", type=str, default="logs/spawn_uav_debug.csv", help="Output path for telemetry CSV.")
parser.add_argument("--log_hz", type=float, default=50.0, help="CSV logging rate in Hz. Set <= 0 to log every step.")
parser.add_argument("--log_env_id", type=int, default=0, help="Environment index to log into CSV.")
parser.add_argument("--live_plot", action="store_true", help="Show runtime telemetry plot window.")
parser.add_argument("--live_plot_hz", type=float, default=10.0, help="Live plot refresh rate in Hz.")
parser.add_argument("--live_plot_window_s", type=float, default=20.0, help="Live plot rolling window in seconds.")
parser.add_argument("--live_plot_env_id", type=int, default=0, help="Environment index to plot at runtime.")
parser.add_argument("--follow_cam", action="store_true", help="Enable smoothed chase camera.")
parser.add_argument("--follow_cam_env_id", type=int, default=0, help="Environment index for chase camera.")
parser.add_argument("--follow_cam_distance", type=float, default=10.0, help="Chase distance behind UAV [m].")
parser.add_argument("--follow_cam_height", type=float, default=3.0, help="Chase camera height above UAV [m].")
parser.add_argument("--follow_cam_lookahead", type=float, default=6.0, help="Look-ahead distance in front of UAV [m].")
parser.add_argument("--follow_cam_smooth_tau", type=float, default=0.35, help="Smoothing time constant for chase camera [s].")
parser.add_argument(
    "--thrust_pos",
    type=float,
    nargs=3,
    default=None,
    help="Thrust application point in body frame (x y z). Default: body COM (no parasitic moment).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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

# Aero presets for quick A/B testing.
# Switch by assigning one of these to AERO_CFG below.
AERO_CFG_BASELINE = AeroConfig()

# Softer lateral weathercock/roll response from sideslip.
AERO_CFG_SOFT_LATERAL = AeroConfig(
    CY_beta=-0.75,
    Cl_beta=-0.08,
    Cn_beta=0.18,
)

# Stronger yaw damping, gentler roll/yaw rate coupling.
AERO_CFG_DAMPED_RATES = AeroConfig(
    Cl_p=-0.35,
    Cl_r=0.08,
    Cn_p=-0.03,
    Cn_r=-0.30,
)

# Most conservative for ground-run/transient veer suppression.
AERO_CFG_STABLE_GROUND_RUN = AeroConfig(
    CY_beta=-0.65,
    Cl_beta=-0.06,
    Cn_beta=0.15,
    Cl_p=-0.30,
    Cl_r=0.05,
    Cn_p=-0.02,
    Cn_r=-0.35,
)

# Combined compromise between baseline authority and damping.
AERO_CFG_BALANCED = AeroConfig(
    CY_beta=-0.78,
    Cl_beta=-0.09,
    Cn_beta=0.20,
    Cl_p=-0.38,
    Cl_r=0.10,
    Cn_p=-0.04,
    Cn_r=-0.27,
)

AERO_CFG = AERO_CFG_BASELINE


class ManualFlightControlWindow:
    def __init__(self, thrust=0.0, roll=0.0, pitch=0.0, yaw=0.0, max_thrust=50.0, max_torque=10.0):
        import omni.ui as ui

        self._window = ui.Window("UAV Manual Controls", width=360, height=300)
        self._models = {
            "thrust": ui.SimpleFloatModel(thrust),
            "roll": ui.SimpleFloatModel(roll),
            "pitch": ui.SimpleFloatModel(pitch),
            "yaw": ui.SimpleFloatModel(yaw),
            "max_thrust": ui.SimpleFloatModel(max_thrust),
            "max_torque": ui.SimpleFloatModel(max_torque),
        }
        self._build_ui(ui)

    def _build_ui(self, ui):
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Thrust", height=20)
                self._create_drag(ui, "Thrust [0..1]", self._models["thrust"], 0.0, 1.0, 0.01)
                ui.Separator(height=2)
                ui.Label("Attitude Commands", height=20)
                self._create_drag(ui, "Roll [-1..1]", self._models["roll"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Pitch [-1..1]", self._models["pitch"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Yaw [-1..1]", self._models["yaw"], -1.0, 1.0, 0.01)
                ui.Spacer(height=6)
                ui.Separator(height=2)
                ui.Label("Force Limits", height=20)
                self._create_drag(ui, "Max Thrust [N]", self._models["max_thrust"], 0.0, 500.0, 1.0)
                self._create_drag(ui, "Max Torque [N*m]", self._models["max_torque"], 0.0, 200.0, 0.5)
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
            float(self._models["max_torque"].as_float),
        )


def _quat_from_two_vectors(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    v0_n = math_utils.normalize(v0)
    v1_n = math_utils.normalize(v1)
    dot = torch.sum(v0_n * v1_n, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.cross(v0_n, v1_n, dim=-1)
    w = 1.0 + dot
    quat = torch.cat([w, cross], dim=-1)
    # Handle near-opposite vectors
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
        prim_path="/World/Visuals/uavMarkers",
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


def _calculate_propeller_thrust(throttle: torch.Tensor, max_thrust: float) -> torch.Tensor:
    return calculate_propeller_thrust(throttle, max_thrust, PROP_CFG)


class LiveTelemetryPlotter:
    def __init__(self, update_hz: float = 10.0, window_s: float = 20.0):
        import matplotlib.pyplot as plt

        self._plt = plt
        self._update_interval = (1.0 / update_hz) if update_hz > 0.0 else 0.0
        self._window_s = max(0.0, float(window_s))
        self._last_update_t = -1.0e9

        self._series_keys = [
            "thrust_cmd",
            "roll_cmd",
            "pitch_cmd",
            "yaw_cmd",
            "total_force_x",
            "total_force_y",
            "total_force_z",
            "total_torque_x",
            "total_torque_y",
            "total_torque_z",
            "alpha_rad",
            "beta_rad",
            "w_body_x",
            "w_body_y",
            "w_body_z",
        ]
        self._time = deque()
        self._series = {k: deque() for k in self._series_keys}

        self._plt.ion()
        self._fig, self._axes = self._plt.subplots(4, 1, sharex=True, figsize=(10.0, 8.0), constrained_layout=True)
        manager = getattr(self._fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("UAV Live Telemetry")

        self._line_map = {}
        self._line_map["thrust_cmd"] = self._axes[0].plot([], [], label="thrust_cmd")[0]
        self._line_map["roll_cmd"] = self._axes[0].plot([], [], label="roll_cmd")[0]
        self._line_map["pitch_cmd"] = self._axes[0].plot([], [], label="pitch_cmd")[0]
        self._line_map["yaw_cmd"] = self._axes[0].plot([], [], label="yaw_cmd")[0]
        self._axes[0].set_ylabel("Commands")

        self._line_map["total_force_x"] = self._axes[1].plot([], [], label="Fx")[0]
        self._line_map["total_force_y"] = self._axes[1].plot([], [], label="Fy")[0]
        self._line_map["total_force_z"] = self._axes[1].plot([], [], label="Fz")[0]
        self._axes[1].set_ylabel("Force [N]")

        self._line_map["total_torque_x"] = self._axes[2].plot([], [], label="Mx")[0]
        self._line_map["total_torque_y"] = self._axes[2].plot([], [], label="My")[0]
        self._line_map["total_torque_z"] = self._axes[2].plot([], [], label="Mz")[0]
        self._axes[2].set_ylabel("Torque [N*m]")

        self._line_map["alpha_rad"] = self._axes[3].plot([], [], label="alpha")[0]
        self._line_map["beta_rad"] = self._axes[3].plot([], [], label="beta")[0]
        self._line_map["w_body_x"] = self._axes[3].plot([], [], label="p")[0]
        self._line_map["w_body_y"] = self._axes[3].plot([], [], label="q")[0]
        self._line_map["w_body_z"] = self._axes[3].plot([], [], label="r")[0]
        self._axes[3].set_ylabel("Aero/Rate")
        self._axes[3].set_xlabel("Time [s]")

        for ax in self._axes:
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")

    def add_sample(self, t: float, sample: dict[str, float]):
        self._time.append(float(t))
        for key in self._series_keys:
            self._series[key].append(float(sample.get(key, float("nan"))))

        if self._window_s > 0.0:
            min_time = float(t) - self._window_s
            while self._time and self._time[0] < min_time:
                self._time.popleft()
                for key in self._series_keys:
                    self._series[key].popleft()

        if self._update_interval == 0.0 or (t - self._last_update_t) >= self._update_interval:
            self._redraw()
            self._last_update_t = float(t)

    def _redraw(self):
        if not self._time:
            return

        x = list(self._time)
        for key, line in self._line_map.items():
            line.set_data(x, list(self._series[key]))

        for ax in self._axes:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        if self._window_s > 0.0:
            x_max = x[-1]
            x_min = max(0.0, x_max - self._window_s)
        else:
            x_min = x[0]
            x_max = x[-1]
        if x_max <= x_min:
            x_max = x_min + 1.0e-3
        for ax in self._axes:
            ax.set_xlim(x_min, x_max)

        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def close(self):
        self._plt.close(self._fig)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, markers: VisualizationMarkers | None):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    uav = scene["uav"]
    body_ids, _ = uav.find_bodies("body")
    if len(body_ids) == 0:
        body_ids = [0]

    device = uav.device
    body_ids_t = torch.tensor(body_ids, device=device, dtype=torch.int32)
    num_envs = uav.num_instances

    thrust_cmd = float(args_cli.thrust)
    roll_cmd = float(args_cli.roll)
    pitch_cmd = float(args_cli.pitch)
    yaw_cmd = float(args_cli.yaw)

    max_thrust = float(args_cli.max_thrust)
    max_torque = float(args_cli.max_torque)
    debug_print_hz = float(args_cli.debug_print_hz)
    debug_print_interval = (1.0 / debug_print_hz) if debug_print_hz > 0.0 else None
    last_debug_print_time = -1.0e9
    last_debug_line_len = 0
    thrust_pos_override = (
        torch.tensor(args_cli.thrust_pos, device=device, dtype=torch.float32) if args_cli.thrust_pos is not None else None
    )
    if thrust_pos_override is None:
        print("[INFO]: Applying thrust at body COM (default). Use --thrust_pos x y z to override.")
    else:
        print(f"[INFO]: Applying thrust at body-frame position: {tuple(args_cli.thrust_pos)}")

    log_writer = None
    log_file = None
    log_interval = (1.0 / float(args_cli.log_hz)) if float(args_cli.log_hz) > 0.0 else 0.0
    last_log_time = -1.0e9
    log_env_id = int(max(0, min(num_envs - 1, int(args_cli.log_env_id)))) if num_envs > 0 else 0
    if args_cli.log_csv:
        log_dir = os.path.dirname(args_cli.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(args_cli.log_path, "w", newline="", buffering=1)
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "sim_time_s",
                "step",
                "env_id",
                "thrust_cmd",
                "roll_cmd",
                "pitch_cmd",
                "yaw_cmd",
                "max_thrust",
                "max_torque",
                "pos_w_x",
                "pos_w_y",
                "pos_w_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "v_body_x",
                "v_body_y",
                "v_body_z",
                "w_body_x",
                "w_body_y",
                "w_body_z",
                "speed_mps",
                "alpha_rad",
                "beta_rad",
                "thrust_force_x",
                "thrust_force_y",
                "thrust_force_z",
                "aero_force_x",
                "aero_force_y",
                "aero_force_z",
                "drag_force_x",
                "drag_force_y",
                "drag_force_z",
                "total_force_x",
                "total_force_y",
                "total_force_z",
                "control_torque_x",
                "control_torque_y",
                "control_torque_z",
                "aero_moment_x",
                "aero_moment_y",
                "aero_moment_z",
                "total_torque_x",
                "total_torque_y",
                "total_torque_z",
                "com_pos_b_x",
                "com_pos_b_y",
                "com_pos_b_z",
                "thrust_pos_b_x",
                "thrust_pos_b_y",
                "thrust_pos_b_z",
                "thrust_lever_arm_b_x",
                "thrust_lever_arm_b_y",
                "thrust_lever_arm_b_z",
            ]
        )
        print(f"[INFO]: Logging telemetry to CSV: {args_cli.log_path} (env {log_env_id}, {args_cli.log_hz} Hz)")

    live_plotter = None
    plot_env_id = int(max(0, min(num_envs - 1, int(args_cli.live_plot_env_id)))) if num_envs > 0 else 0
    if args_cli.live_plot:
        try:
            live_plotter = LiveTelemetryPlotter(
                update_hz=float(args_cli.live_plot_hz),
                window_s=float(args_cli.live_plot_window_s),
            )
            print(
                "[INFO]: Live plotting enabled "
                f"(env {plot_env_id}, refresh {args_cli.live_plot_hz} Hz, window {args_cli.live_plot_window_s} s)."
            )
        except Exception as e:
            print(f"[WARN]: Failed to initialize live plotter: {e}")
            live_plotter = None

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

    def _vec3(x: torch.Tensor, env_idx: int) -> tuple[float, float, float]:
        v = x[env_idx]
        return float(v[0].item()), float(v[1].item()), float(v[2].item())

    control_ui = None
    if args_cli.ui:
        try:
            control_ui = ManualFlightControlWindow(
                thrust=thrust_cmd,
                roll=roll_cmd,
                pitch=pitch_cmd,
                yaw=yaw_cmd,
                max_thrust=max_thrust,
                max_torque=max_torque,
            )
        except Exception as e:
            print(f"[WARN]: Failed to create UI controls: {e}")
            control_ui = None

    try:
        while simulation_app.is_running():
            if control_ui is not None:
                thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd, max_thrust, max_torque = control_ui.get_inputs()
            state = uav.data.root_state_w
            quat_wxyz = state[:, 3:7]
            v_world = state[:, 7:10]
            w_world = state[:, 10:13]

            v_body = math_utils.quat_apply_inverse(quat_wxyz, v_world)
            w_body = math_utils.quat_apply_inverse(quat_wxyz, w_world)

            thrust = clamp(torch.full((num_envs,), thrust_cmd, device=device), 0.0, 1.0)
            thrust = torch.clamp(CONTROL_SIGNS.throttle * thrust, min=0.0, max=1.0)
            thrust_force = _calculate_propeller_thrust(thrust, max_thrust)

            aileron = clamp(torch.full((num_envs,), roll_cmd, device=device), -1.0, 1.0) * CONTROL_SIGNS.aileron
            elevator = clamp(torch.full((num_envs,), pitch_cmd, device=device), -1.0, 1.0) * CONTROL_SIGNS.elevator
            rudder = clamp(torch.full((num_envs,), yaw_cmd, device=device), -1.0, 1.0) * CONTROL_SIGNS.rudder

            control_torque = torch.stack(
                [
                    aileron * max_torque,
                    elevator * max_torque,
                    rudder * max_torque,
                ],
                dim=-1,
            )

            aero_forces, aero_moments = compute_aero_forces_and_moments(
                v_body, w_body, aileron, elevator, rudder, AERO_CFG
            )
            drag_force = linear_drag(v_body, DRAG_CFG)
            total_forces = clamp(thrust_force + aero_forces + drag_force, -MAX_FORCE, MAX_FORCE)
            total_torques = clamp(control_torque + aero_moments, -MAX_TORQUE, MAX_TORQUE)

            composer = uav.instantaneous_wrench_composer
            thrust_force_clamped = clamp(thrust_force, -MAX_FORCE, MAX_FORCE)
            aero_force_clamped = clamp(aero_forces, -MAX_FORCE, MAX_FORCE)
            drag_force_clamped = clamp(drag_force, -MAX_FORCE, MAX_FORCE)
            total_torques_clamped = clamp(total_torques, -MAX_TORQUE, MAX_TORQUE)

            thrust_positions = torch.zeros((num_envs, len(body_ids), 3), device=device)
            if thrust_pos_override is None:
                thrust_positions[:, 0, :] = uav.data.body_com_pos_b[:, 0]
            else:
                thrust_positions[:, 0, :] = thrust_pos_override
            com_positions = torch.zeros((num_envs, len(body_ids), 3), device=device)
            com_positions[:, 0, :] = uav.data.body_com_pos_b[:, 0]

            if debug_print_interval is not None and (sim_time - last_debug_print_time) >= debug_print_interval:
                f_th = thrust_force[0]
                f_a = aero_forces[0]
                f_d = drag_force[0]
                f_t = total_forces[0]
                t_c = control_torque[0]
                t_a = aero_moments[0]
                t_t = total_torques[0]

                thrust_p = thrust_positions[0, 0]
                aero_p = com_positions[0, 0]
                body_com = uav.data.body_com_pos_b[0, 0]

                line = (
                    f"env0 Fth({f_th[0].item():7.2f},{f_th[1].item():7.2f},{f_th[2].item():7.2f}) "
                    f"Fa({f_a[0].item():7.2f},{f_a[1].item():7.2f},{f_a[2].item():7.2f}) "
                    # f"Fd({f_d[0].item():7.2f},{f_d[1].item():7.2f},{f_d[2].item():7.2f}) "
                    # f"Ft({f_t[0].item():7.2f},{f_t[1].item():7.2f},{f_t[2].item():7.2f}) | "
                    # f"Tc({t_c[0].item():7.2f},{t_c[1].item():7.2f},{t_c[2].item():7.2f}) "
                    f"Ta({t_a[0].item():7.2f},{t_a[1].item():7.2f},{t_a[2].item():7.2f}) "
                    # f"Tt({t_t[0].item():7.2f},{t_t[1].item():7.2f},{t_t[2].item():7.2f}) | "
                    f"thrust@b({thrust_p[0].item():6.3f},{thrust_p[1].item():6.3f},{thrust_p[2].item():6.3f}) "
                    f"aero@b({aero_p[0].item():6.3f},{aero_p[1].item():6.3f},{aero_p[2].item():6.3f}) "
                    f"com@b({body_com[0].item():6.3f},{body_com[1].item():6.3f},{body_com[2].item():6.3f}) "
                    # f"torque_body={body_ids[0]}"
                )
                pad = " " * max(0, last_debug_line_len - len(line))
                print(f"\r{line}{pad}", end="", flush=True)
                last_debug_line_len = len(line)
                last_debug_print_time = sim_time

            if log_writer is not None and (log_interval == 0.0 or (sim_time - last_log_time) >= log_interval):
                env_i = log_env_id
                pos = state[env_i, 0:3]
                quat = state[env_i, 3:7]
                v_b = v_body[env_i]
                speed = torch.linalg.norm(v_b).item()
                safe_speed = max(speed, 0.1)
                alpha = torch.atan2(-v_b[2], v_b[0]).item()
                beta = torch.asin(torch.clamp(-v_b[1] / safe_speed, min=-1.0, max=1.0)).item()

                thrust_p = thrust_positions[env_i, 0]
                com_p = com_positions[env_i, 0]
                lever = thrust_p - com_p

                tf_x, tf_y, tf_z = _vec3(thrust_force, env_i)
                af_x, af_y, af_z = _vec3(aero_forces, env_i)
                df_x, df_y, df_z = _vec3(drag_force, env_i)
                ff_x, ff_y, ff_z = _vec3(total_forces, env_i)
                ct_x, ct_y, ct_z = _vec3(control_torque, env_i)
                am_x, am_y, am_z = _vec3(aero_moments, env_i)
                tt_x, tt_y, tt_z = _vec3(total_torques, env_i)
                vb_x, vb_y, vb_z = _vec3(v_body, env_i)
                wb_x, wb_y, wb_z = _vec3(w_body, env_i)

                log_writer.writerow(
                    [
                        float(sim_time),
                        int(count),
                        int(env_i),
                        float(thrust_cmd),
                        float(roll_cmd),
                        float(pitch_cmd),
                        float(yaw_cmd),
                        float(max_thrust),
                        float(max_torque),
                        float(pos[0].item()),
                        float(pos[1].item()),
                        float(pos[2].item()),
                        float(quat[0].item()),
                        float(quat[1].item()),
                        float(quat[2].item()),
                        float(quat[3].item()),
                        vb_x,
                        vb_y,
                        vb_z,
                        wb_x,
                        wb_y,
                        wb_z,
                        float(speed),
                        float(alpha),
                        float(beta),
                        tf_x,
                        tf_y,
                        tf_z,
                        af_x,
                        af_y,
                        af_z,
                        df_x,
                        df_y,
                        df_z,
                        ff_x,
                        ff_y,
                        ff_z,
                        ct_x,
                        ct_y,
                        ct_z,
                        am_x,
                        am_y,
                        am_z,
                        tt_x,
                        tt_y,
                        tt_z,
                        float(com_p[0].item()),
                        float(com_p[1].item()),
                        float(com_p[2].item()),
                        float(thrust_p[0].item()),
                        float(thrust_p[1].item()),
                        float(thrust_p[2].item()),
                        float(lever[0].item()),
                        float(lever[1].item()),
                        float(lever[2].item()),
                    ]
                )
                last_log_time = sim_time

            if live_plotter is not None:
                env_i = plot_env_id
                v_b = v_body[env_i]
                speed = torch.linalg.norm(v_b).item()
                safe_speed = max(speed, 0.1)
                alpha = torch.atan2(-v_b[2], v_b[0]).item()
                beta = torch.asin(torch.clamp(-v_b[1] / safe_speed, min=-1.0, max=1.0)).item()
                live_plotter.add_sample(
                    sim_time,
                    {
                        "thrust_cmd": float(thrust_cmd),
                        "roll_cmd": float(roll_cmd),
                        "pitch_cmd": float(pitch_cmd),
                        "yaw_cmd": float(yaw_cmd),
                        "total_force_x": float(total_forces[env_i, 0].item()),
                        "total_force_y": float(total_forces[env_i, 1].item()),
                        "total_force_z": float(total_forces[env_i, 2].item()),
                        "total_torque_x": float(total_torques[env_i, 0].item()),
                        "total_torque_y": float(total_torques[env_i, 1].item()),
                        "total_torque_z": float(total_torques[env_i, 2].item()),
                        "alpha_rad": float(alpha),
                        "beta_rad": float(beta),
                        "w_body_x": float(w_body[env_i, 0].item()),
                        "w_body_y": float(w_body[env_i, 1].item()),
                        "w_body_z": float(w_body[env_i, 2].item()),
                    },
                )

            forces_t = torch.zeros((num_envs, len(body_ids), 3), device=device)
            forces_t[:, 0, :] = thrust_force_clamped
            composer.add_forces_and_torques(forces=forces_t, positions=thrust_positions, body_ids=body_ids_t)

            forces_a = torch.zeros((num_envs, len(body_ids), 3), device=device)
            forces_a[:, 0, :] = aero_force_clamped + drag_force_clamped
            composer.add_forces_and_torques(forces=forces_a, positions=com_positions, body_ids=body_ids_t)

            torques = torch.zeros((num_envs, len(body_ids), 3), device=device)
            torques[:, 0, :] = total_torques_clamped
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

            scene.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
            count += 1
            scene.update(sim_dt)

            if follow_cam is not None:
                follow_cam.step(uav.data.root_state_w)
    finally:
        if debug_print_interval is not None and last_debug_line_len > 0:
            print()
        if log_file is not None:
            log_file.close()
            print(f"[INFO]: CSV log saved: {args_cli.log_path}")
        if live_plotter is not None:
            live_plotter.close()


def main():

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Design scene
    scene_cfg = BasicFixedWing1SceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    markers = _create_visual_markers() if args_cli.vis_markers else None

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, markers)


if __name__ == "__main__":
    main()
    simulation_app.close()
