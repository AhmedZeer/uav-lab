import argparse
import math
from dataclasses import replace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-wing scene with a setpoint-driven autopilot and per-surface aerodynamic modeling.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--env_spacing", type=int, default=7, help="Number of environments.")
parser.add_argument("--ui", action="store_true", help="Enable UI controls for thrust and attitude/heading setpoints.")
parser.add_argument("--thrust", type=float, default=0.0, help="Initial throttle setpoint [0, 1].")
parser.add_argument("--roll", type=float, default=0.0, help="Initial roll setpoint [-1, 1].")
parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch setpoint [-1, 1].")
parser.add_argument("--yaw", type=float, default=0.0, help="Initial heading-offset setpoint [-1, 1].")
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
parser.add_argument("--start_roll_deg", type=float, default=0.0, help="Initial roll [deg].")
parser.add_argument("--start_pitch_deg", type=float, default=0.0, help="Initial pitch [deg].")
parser.add_argument("--start_yaw_deg", type=float, default=0.0, help="Initial yaw [deg].")
parser.add_argument(
    "--start_roll_noise_deg",
    type=float,
    default=0.0,
    help="Uniform per-environment roll noise half-range [deg].",
)
parser.add_argument(
    "--start_pitch_noise_deg",
    type=float,
    default=0.0,
    help="Uniform per-environment pitch noise half-range [deg].",
)
parser.add_argument(
    "--start_yaw_noise_deg",
    type=float,
    default=0.0,
    help="Uniform per-environment yaw noise half-range [deg].",
)
parser.add_argument(
    "--start_speed_noise_mps",
    type=float,
    default=0.0,
    help="Uniform per-environment forward-speed noise half-range [m/s].",
)
parser.add_argument(
    "--start_body_rate_noise_rps",
    type=float,
    default=0.0,
    help="Uniform per-environment body-rate noise half-range [rad/s].",
)

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

import carb
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils
from isaacsim.util.debug_draw import _debug_draw
from uav_lab_1.controllers import FixedWingAutopilot, FixedWingAutopilotConfig
from uav_lab_1.robots.follow_camera import FollowCameraConfig, SmoothedFollowCamera

# from uav_lab_1.robots.surface_aero import SurfaceAeroModel, default_fixedwing_surface_configs
from uav_lab_1.robots.aero import compute_aero_forces_and_moments, AeroConfig

from uav_lab_1.scenes import BasicFixedWing1SceneCfg, MudFixedWing1SceneCfg, MudCubeSceneCfg

MAX_FORCE = 1000.0
MAX_TORQUE = 1000.0


class ManualSurfaceControlWindow:
    def __init__(self, thrust=0.0, roll=0.0, pitch=0.0, yaw=0.0, max_thrust=100.0):
        import omni.ui as ui

        self._window = ui.Window("Fixed-Wing Autopilot Setpoints", width=380, height=280)
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
                ui.Label("Autopilot Setpoints", height=20)
                self._create_drag(ui, "Throttle SP [0..1]", self._models["thrust"], 0.0, 1.0, 0.01)
                self._create_drag(ui, "Max Thrust [N]", self._models["max_thrust"], 0.0, 500.0, 1.0)
                ui.Separator(height=2)
                ui.Label("Normalized Setpoints", height=20)
                self._create_drag(ui, "Roll SP [-1..1]", self._models["roll"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Pitch SP [-1..1]", self._models["pitch"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Heading SP [-1..1]", self._models["yaw"], -1.0, 1.0, 0.01)
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


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def _quat_from_yaw_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _quat_from_rpy_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


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


def _sample_symmetric_noise(num_envs: int, magnitude: float, device: torch.device) -> torch.Tensor:
    if magnitude <= 0.0:
        return torch.zeros((num_envs,), device=device, dtype=torch.float32)
    return (2.0 * torch.rand((num_envs,), device=device, dtype=torch.float32) - 1.0) * float(magnitude)


def _init_airborne_state(
    uav,
    scene: InteractiveScene,
    altitude: float,
    speed: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    roll_noise_deg: float = 0.0,
    pitch_noise_deg: float = 0.0,
    yaw_noise_deg: float = 0.0,
    speed_noise_mps: float = 0.0,
    body_rate_noise_rps: float = 0.0,
):
    device = uav.device
    num_envs = uav.num_instances
    env_ids = torch.arange(num_envs, device=device, dtype=torch.int64)

    default_root_state = uav.data.default_root_state.clone()
    root_pose = default_root_state[:, :7].clone()
    root_vel = default_root_state[:, 7:].clone()

    root_pose[:, :3] = scene.env_origins.to(device) + torch.tensor([0.0, 0.0, altitude], device=device)
    roll = torch.full((num_envs,), math.radians(roll_deg), device=device, dtype=torch.float32)
    pitch = torch.full((num_envs,), math.radians(pitch_deg), device=device, dtype=torch.float32)
    yaw = torch.full((num_envs,), math.radians(yaw_deg), device=device, dtype=torch.float32)

    roll = roll + torch.deg2rad(_sample_symmetric_noise(num_envs, roll_noise_deg, device))
    pitch = pitch + torch.deg2rad(_sample_symmetric_noise(num_envs, pitch_noise_deg, device))
    yaw = yaw + torch.deg2rad(_sample_symmetric_noise(num_envs, yaw_noise_deg, device))
    quat_wxyz = _quat_from_rpy_wxyz(roll, pitch, yaw)
    root_pose[:, 3:7] = quat_wxyz

    speed_body = torch.clamp(
        torch.full((num_envs,), speed, device=device, dtype=torch.float32)
        + _sample_symmetric_noise(num_envs, speed_noise_mps, device),
        min=0.0,
    )
    lin_vel_body = torch.stack(
        [speed_body, torch.zeros_like(speed_body), torch.zeros_like(speed_body)],
        dim=-1,
    )
    ang_vel_body = torch.stack(
        [
            _sample_symmetric_noise(num_envs, body_rate_noise_rps, device),
            _sample_symmetric_noise(num_envs, body_rate_noise_rps, device),
            _sample_symmetric_noise(num_envs, body_rate_noise_rps, device),
        ],
        dim=-1,
    )
    root_vel[:, 0:3] = math_utils.quat_apply(quat_wxyz, lin_vel_body)
    root_vel[:, 3:6] = math_utils.quat_apply(quat_wxyz, ang_vel_body)

    uav.write_root_pose_to_sim(root_pose, env_ids)
    uav.write_root_velocity_to_sim(root_vel, env_ids)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()

    uav = scene["uav"]
    body_ids, _ = uav.find_bodies("body")
    if len(body_ids) == 0:
        body_ids = [0]
    body_id = int(body_ids[0])
    propeller_joint_ids, _ = uav.find_joints("propeller_joint")
    propeller_joint_ids = [int(j) for j in propeller_joint_ids]

    device = uav.device
    num_envs = uav.num_instances
    body_ids_t = torch.tensor([body_id], device=device, dtype=torch.int32)

    thrust_cmd = float(args_cli.thrust)
    roll_cmd = float(args_cli.roll)
    pitch_cmd = float(args_cli.pitch)
    yaw_cmd = float(args_cli.yaw)
    max_thrust = float(args_cli.max_thrust)

    control_ui = ManualSurfaceControlWindow(
        thrust=thrust_cmd,
        roll=roll_cmd,
        pitch=pitch_cmd,
        yaw=yaw_cmd,
        max_thrust=max_thrust,
    )

    aero_cfg = AeroConfig()
    # draw = _debug_draw.acquire_debug_draw_interface()
    # force_debug_scale = 0.02
    autopilot_cfg = FixedWingAutopilotConfig(dt=sim_dt)
    autopilot = FixedWingAutopilot(autopilot_cfg, num_envs=num_envs, device=device)

    throttle_state = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    thrust_tau = max(1.0e-4, float(args_cli.thrust_tau))
    propeller_effort_target = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)

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


    while simulation_app.is_running():
        thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd, max_thrust = control_ui.get_inputs()

        state = uav.data.root_state_w
        pos_world = state[:, 0:3]
        quat_wxyz = state[:, 3:7]
        v_world = state[:, 7:10]
        w_world = state[:, 10:13]
        # body_com_b = torch.zeros_like(uav.data.body_com_pos_b[:, body_id])
        body_com_b = uav.data.body_com_pos_b[:, body_id]

        v_body = math_utils.quat_apply_inverse(quat_wxyz, v_world)
        w_body = math_utils.quat_apply_inverse(quat_wxyz, w_world)
        roll, pitch, yaw = _euler_from_quat_wxyz(quat_wxyz)
        # print(f"CoM_b : {body_com_b[0]}")
        # print(f"pos_w : {pos_world[0]}")
        # print(f"RPY : {yaw[0]:.2f}, {pitch[0]:.2f}, {roll[0]:.2f}")

        control_out = autopilot.step(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            v_body=v_body,
            w_body=w_body,
            thrust_setpoint=torch.full((num_envs,), thrust_cmd, device=device, dtype=torch.float32),
            roll_setpoint_norm=torch.full((num_envs,), roll_cmd, device=device, dtype=torch.float32),
            pitch_setpoint_norm=torch.full((num_envs,), pitch_cmd, device=device, dtype=torch.float32),
            yaw_setpoint_norm=torch.full((num_envs,), yaw_cmd, device=device, dtype=torch.float32),
        )
        thrust_cmd_t = control_out["throttle_cmd"]
        # thrust_cmd_t = thrust_cmd

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
        else:
            propeller_effort_target.zero_()

        thrust_force = torch.stack(
            [
                throttle_state * float(max_thrust),
                torch.zeros_like(throttle_state),
                torch.zeros_like(throttle_state),
            ],
            dim=-1,
        )


        print(
            f"Pilot Cmds: {control_out['aileron_cmd'][0]:.2f}, {control_out['elevator_cmd'][0]:.2f}, {control_out['rudder_cmd'][0]:.2f}\n"
        )
        print(
            f"Pilot Errors: {control_out['roll_error'][0]:.3f}, {control_out['pitch_error'][0]:.2f}, {control_out['heading_error'][0]:.3f}, {control_out['sideslip_error'][0]:.3f}\n"
        )
        forces_flu, moments_flu = compute_aero_forces_and_moments(
            v_body=v_body,
            w_body=w_body,
            aileron=control_out['aileron_cmd'],
            elevator=control_out['elevator_cmd'],
            rudder=control_out['rudder_cmd'],
            cfg=aero_cfg,
        )

        total_forces = _clamp(thrust_force + forces_flu, -MAX_FORCE, MAX_FORCE)
        total_torques = _clamp(moments_flu, -MAX_FORCE, MAX_FORCE)
        # print(f"{total_forces=}")
        # print(f"{total_torques=}")

        # force_world = math_utils.quat_apply(quat_wxyz, total_forces)
        # draw.clear_lines()
        # force_start = pos_world[0]
        # force_end = force_start + (force_debug_scale * force_world[0])
        # draw.draw_lines(
        #     [carb.Float3(float(force_start[0].item()), float(force_start[1].item()), float(force_start[2].item()))],
        #     [carb.Float3(float(force_end[0].item()), float(force_end[1].item()), float(force_end[2].item()))],
        #     [(1.0, 0.3, 0.0, 1.0)],
        #     [5.0],
        # )

        com_positions = torch.zeros((num_envs, 1, 3), device=device)
        com_positions[:, 0, :] = body_com_b

        forces = torch.zeros((num_envs, 1, 3), device=device)
        forces[:, 0, :] = total_forces

        torques = torch.zeros((num_envs, 1, 3), device=device)
        torques[:, 0, :] = total_torques

        composer = uav.instantaneous_wrench_composer
        composer.add_forces_and_torques(forces=forces, positions=com_positions, body_ids=body_ids_t)
        composer.add_forces_and_torques(torques=torques, body_ids=body_ids_t)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if follow_cam is not None:
            follow_cam.step(uav.data.root_state_w)


def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-15.0, -8.0, 8.0], [0.0, 0.0, args_cli.start_alt])

    scene_cfg = MudFixedWing1SceneCfg(args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    uav = scene["uav"]
    _init_airborne_state(
        uav=uav,
        scene=scene,
        altitude=float(args_cli.start_alt),
        speed=float(args_cli.start_speed),
        roll_deg=float(args_cli.start_roll_deg),
        pitch_deg=float(args_cli.start_pitch_deg),
        yaw_deg=float(args_cli.start_yaw_deg),
        roll_noise_deg=float(args_cli.start_roll_noise_deg),
        pitch_noise_deg=float(args_cli.start_pitch_noise_deg),
        yaw_noise_deg=float(args_cli.start_yaw_noise_deg),
        speed_noise_mps=float(args_cli.start_speed_noise_mps),
        body_rate_noise_rps=float(args_cli.start_body_rate_noise_rps),
    )
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    print("[INFO]: Airborne init complete. Running fixed-wing autopilot scene.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
