import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Fixed-wing scene with multiple UAVs spawned in an inverted-V formation."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned environments.")
parser.add_argument("--num_uavs", type=int, default=5, help="Number of UAVs to spawn per environment.")
parser.add_argument("--env_spacing", type=float, default=500.0, help="Spacing between cloned environments [m].")
parser.add_argument("--ui", action="store_true", help="Enable UI controls for common setpoints.")
parser.add_argument("--thrust", type=float, default=0.55, help="Common throttle setpoint [0, 1].")
parser.add_argument("--roll", type=float, default=0.0, help="Common roll setpoint [-1, 1].")
parser.add_argument("--pitch", type=float, default=0.0, help="Common pitch setpoint [-1, 1].")
parser.add_argument("--yaw", type=float, default=0.0, help="Common heading-offset setpoint [-1, 1].")
parser.add_argument("--max_thrust", type=float, default=200.0, help="Maximum thrust [N].")
parser.add_argument("--thrust_tau", type=float, default=0.02, help="First-order thrust lag time constant [s].")
parser.add_argument("--start_alt", type=float, default=100.0, help="Initial altitude [m].")
parser.add_argument("--start_speed", type=float, default=40.0, help="Initial forward speed [m/s].")
parser.add_argument("--start_roll_deg", type=float, default=0.0, help="Initial roll [deg].")
parser.add_argument("--start_pitch_deg", type=float, default=0.0, help="Initial pitch [deg].")
parser.add_argument("--start_yaw_deg", type=float, default=0.0, help="Initial formation heading/yaw [deg].")
parser.add_argument(
    "--longitudinal_spacing",
    type=float,
    default=35.0,
    help="Backward spacing between inverted-V rows [m].",
)
parser.add_argument(
    "--lateral_spacing",
    type=float,
    default=25.0,
    help="Side spacing between inverted-V rows [m].",
)
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
parser.add_argument("--follow_cam", action="store_true", help="Enable smoothed chase camera.")
parser.add_argument("--follow_cam_uav_id", type=int, default=0, help="UAV index for chase camera.")
parser.add_argument("--follow_cam_env_id", type=int, default=0, help="Environment index for chase camera.")
parser.add_argument("--follow_cam_distance", type=float, default=35.0, help="Chase distance behind UAV [m].")
parser.add_argument("--follow_cam_height", type=float, default=12.0, help="Chase camera height above UAV [m].")
parser.add_argument("--follow_cam_lookahead", type=float, default=18.0, help="Look-ahead distance in front of UAV [m].")
parser.add_argument("--follow_cam_smooth_tau", type=float, default=0.2, help="Smoothing time constant [s].")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

import uav_lab_1.scenes as uav_scenes
from uav_lab_1.controllers import FixedWingAutopilot, FixedWingAutopilotConfig
from uav_lab_1.robots.aero import AeroConfig, compute_aero_forces_and_moments
from uav_lab_1.robots.fixedwing_1 import FIXEDWING_1_CONFIG
from uav_lab_1.robots.follow_camera import FollowCameraConfig, SmoothedFollowCamera

MAX_FORCE = 1000.0
MAX_TORQUE = 1000.0


@configclass
class MultiFixedWingSceneCfg(InteractiveSceneCfg):
    """Scene with puddle ground. UAV assets are added dynamically after construction."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(uav_scenes.__file__), "usd", "puddles", "Puddles.usd")
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


class ManualSurfaceControlWindow:
    def __init__(self, thrust=0.0, roll=0.0, pitch=0.0, yaw=0.0, max_thrust=100.0):
        import omni.ui as ui

        self._window = ui.Window("Multi-UAV Common Setpoints", width=380, height=280)
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
                ui.Label("Common Autopilot Setpoints", height=20)
                self._create_drag(ui, "Throttle SP [0..1]", self._models["thrust"], 0.0, 1.0, 0.01)
                self._create_drag(ui, "Max Thrust [N]", self._models["max_thrust"], 0.0, 500.0, 1.0)
                ui.Separator(height=2)
                self._create_drag(ui, "Roll SP [-1..1]", self._models["roll"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Pitch SP [-1..1]", self._models["pitch"], -1.0, 1.0, 0.01)
                self._create_drag(ui, "Heading SP [-1..1]", self._models["yaw"], -1.0, 1.0, 0.01)
                ui.Spacer(height=6)
                ui.Button("Zero Attitude", clicked_fn=self.zero_attitude)

    def _create_drag(self, ui, label, model, min_val, max_val, step):
        with ui.HStack(height=24):
            ui.Label(label, width=130)
            ui.FloatDrag(model, min=min_val, max=max_val, step=step)

    def zero_attitude(self):
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


class FollowCameraControlWindow:
    def __init__(
        self,
        *,
        num_uavs: int,
        num_envs: int,
        uav_id: int,
        env_id: int,
        distance: float,
        height: float,
        lookahead: float,
        smooth_tau: float,
    ):
        import omni.ui as ui

        self.num_uavs = int(num_uavs)
        self.num_envs = int(num_envs)
        self._window = ui.Window("Follow Camera Controls", width=390, height=300)
        self._models = {
            "uav_id": ui.SimpleIntModel(uav_id),
            "env_id": ui.SimpleIntModel(env_id),
            "distance": ui.SimpleFloatModel(distance),
            "height": ui.SimpleFloatModel(height),
            "lookahead": ui.SimpleFloatModel(lookahead),
            "smooth_tau": ui.SimpleFloatModel(smooth_tau),
        }
        self._reset_requested = False
        self._build_ui(ui)

    def _build_ui(self, ui):
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Follow Camera", height=20)
                self._create_int_drag(ui, "UAV ID", self._models["uav_id"], 0, max(0, self.num_uavs - 1), 1)
                self._create_int_drag(ui, "Env ID", self._models["env_id"], 0, max(0, self.num_envs - 1), 1)
                ui.Separator(height=2)
                self._create_float_drag(ui, "Distance [m]", self._models["distance"], -250.0, 250.0, 1.0)
                self._create_float_drag(ui, "Height [m]", self._models["height"], -50.0, 150.0, 1.0)
                self._create_float_drag(ui, "Lookahead [m]", self._models["lookahead"], -250.0, 250.0, 1.0)
                self._create_float_drag(ui, "Smooth Tau [s]", self._models["smooth_tau"], 0.0, 5.0, 0.01)
                ui.Spacer(height=6)
                ui.Button("Reset Camera Smoothing", clicked_fn=self.request_reset)

    def _create_float_drag(self, ui, label, model, min_val, max_val, step):
        with ui.HStack(height=24):
            ui.Label(label, width=130)
            ui.FloatDrag(model, min=min_val, max=max_val, step=step)

    def _create_int_drag(self, ui, label, model, min_val, max_val, step):
        with ui.HStack(height=24):
            ui.Label(label, width=130)
            ui.IntDrag(model, min=min_val, max=max_val, step=step)

    def request_reset(self):
        self._reset_requested = True

    def get_inputs(self):
        reset_requested = self._reset_requested
        self._reset_requested = False
        return (
            int(max(0, min(self.num_uavs - 1, self._models["uav_id"].as_int))),
            int(max(0, min(self.num_envs - 1, self._models["env_id"].as_int))),
            float(self._models["distance"].as_float),
            float(self._models["height"].as_float),
            float(self._models["lookahead"].as_float),
            float(self._models["smooth_tau"].as_float),
            reset_requested,
        )


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


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


def _inverted_v_offsets(num_uavs: int, longitudinal_spacing: float, lateral_spacing: float, device) -> torch.Tensor:
    offsets = []
    row = 0
    while len(offsets) < num_uavs:
        if row == 0:
            offsets.append((0.0, 0.0, 0.0))
        else:
            offsets.append((-row * longitudinal_spacing, -row * lateral_spacing, 0.0))
            if len(offsets) < num_uavs:
                offsets.append((-row * longitudinal_spacing, row * lateral_spacing, 0.0))
        row += 1
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def _rotate_offsets_by_yaw(offsets: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x = offsets[:, 0].unsqueeze(0)
    y = offsets[:, 1].unsqueeze(0)
    rotated = torch.zeros((yaw.shape[0], offsets.shape[0], 3), device=offsets.device, dtype=torch.float32)
    rotated[:, :, 0] = x * cos_yaw.unsqueeze(-1) - y * sin_yaw.unsqueeze(-1)
    rotated[:, :, 1] = x * sin_yaw.unsqueeze(-1) + y * cos_yaw.unsqueeze(-1)
    rotated[:, :, 2] = offsets[:, 2].unsqueeze(0)
    return rotated


def _make_scene_cfg(num_envs: int, env_spacing: float, num_uavs: int) -> MultiFixedWingSceneCfg:
    scene_cfg = MultiFixedWingSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
    for uav_id in range(num_uavs):
        setattr(scene_cfg, f"uav_{uav_id}", FIXEDWING_1_CONFIG.replace(prim_path=f"{{ENV_REGEX_NS}}/uav_{uav_id}"))
    return scene_cfg


def _init_formation_state(
    uavs: list,
    scene: InteractiveScene,
    altitude: float,
    speed: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    longitudinal_spacing: float,
    lateral_spacing: float,
):
    device = uavs[0].device
    num_envs = uavs[0].num_instances
    env_ids = torch.arange(num_envs, device=device, dtype=torch.int64)

    roll = torch.full((num_envs,), math.radians(roll_deg), device=device, dtype=torch.float32)
    pitch = torch.full((num_envs,), math.radians(pitch_deg), device=device, dtype=torch.float32)
    yaw = torch.full((num_envs,), math.radians(yaw_deg), device=device, dtype=torch.float32)
    quat_wxyz = _quat_from_rpy_wxyz(roll, pitch, yaw)

    offsets = _inverted_v_offsets(len(uavs), longitudinal_spacing, lateral_spacing, device)
    rotated_offsets = _rotate_offsets_by_yaw(offsets, yaw)
    origins = scene.env_origins.to(device)

    lin_vel_body = torch.stack(
        [
            torch.full((num_envs,), speed, device=device, dtype=torch.float32),
            torch.zeros((num_envs,), device=device, dtype=torch.float32),
            torch.zeros((num_envs,), device=device, dtype=torch.float32),
        ],
        dim=-1,
    )
    lin_vel_world = math_utils.quat_apply(quat_wxyz, lin_vel_body)

    for uav_id, uav in enumerate(uavs):
        default_root_state = uav.data.default_root_state.clone()
        root_pose = default_root_state[:, :7].clone()
        root_vel = default_root_state[:, 7:].clone()

        root_pose[:, :3] = origins + rotated_offsets[:, uav_id, :] + torch.tensor(
            [0.0, 0.0, altitude], device=device
        )
        root_pose[:, 3:7] = quat_wxyz
        root_vel[:, 0:3] = lin_vel_world
        root_vel[:, 3:6] = 0.0

        uav.write_root_pose_to_sim(root_pose, env_ids)
        uav.write_root_velocity_to_sim(root_vel, env_ids)

    print("[INFO]: Inverted-V formation offsets [m] in formation frame:")
    for uav_id, offset in enumerate(offsets.detach().cpu().tolist()):
        print(f"  uav_{uav_id}: x={offset[0]:.1f}, y={offset[1]:.1f}, z={offset[2]:.1f}")


def _collect_uav_runtime(uavs: list, sim_dt: float):
    runtime = []
    for uav in uavs:
        body_ids, _ = uav.find_bodies("body")
        if len(body_ids) == 0:
            body_ids = [0]
        body_id = int(body_ids[0])

        propeller_joint_ids, _ = uav.find_joints("propeller_joint")
        propeller_joint_ids = [int(j) for j in propeller_joint_ids]

        runtime.append(
            {
                "body_id": body_id,
                "body_ids_t": torch.tensor([body_id], device=uav.device, dtype=torch.int32),
                "propeller_joint_ids": propeller_joint_ids,
                "autopilot": FixedWingAutopilot(
                    FixedWingAutopilotConfig(dt=sim_dt), num_envs=uav.num_instances, device=uav.device
                ),
                "throttle_state": torch.zeros((uav.num_instances,), device=uav.device, dtype=torch.float32),
                "propeller_effort_target": torch.zeros((uav.num_instances, 1), device=uav.device, dtype=torch.float32),
            }
        )
    return runtime


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, uavs: list):
    sim_dt = sim.get_physics_dt()
    device = uavs[0].device
    num_envs = uavs[0].num_instances

    runtime = _collect_uav_runtime(uavs, sim_dt)
    aero_cfg = AeroConfig()
    thrust_tau = max(1.0e-4, float(args_cli.thrust_tau))

    control_ui = None
    if args_cli.ui:
        control_ui = ManualSurfaceControlWindow(
            thrust=float(args_cli.thrust),
            roll=float(args_cli.roll),
            pitch=float(args_cli.pitch),
            yaw=float(args_cli.yaw),
            max_thrust=float(args_cli.max_thrust),
        )

    follow_cam = None
    follow_cam_ui = None
    if args_cli.follow_cam:
        follow_uav_id = max(0, min(len(uavs) - 1, int(args_cli.follow_cam_uav_id)))
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
        follow_cam_ui = FollowCameraControlWindow(
            num_uavs=len(uavs),
            num_envs=num_envs,
            uav_id=follow_uav_id,
            env_id=follow_cam.env_id,
            distance=float(args_cli.follow_cam_distance),
            height=float(args_cli.follow_cam_height),
            lookahead=float(args_cli.follow_cam_lookahead),
            smooth_tau=float(args_cli.follow_cam_smooth_tau),
        )
        print(f"[INFO]: Smoothed follow camera tracking uav_{follow_uav_id}, env {follow_cam.env_id}.")
    else:
        follow_uav_id = 0

    while simulation_app.is_running():
        if control_ui is not None:
            thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd, max_thrust = control_ui.get_inputs()
        else:
            thrust_cmd = float(args_cli.thrust)
            roll_cmd = float(args_cli.roll)
            pitch_cmd = float(args_cli.pitch)
            yaw_cmd = float(args_cli.yaw)
            max_thrust = float(args_cli.max_thrust)

        for uav, data in zip(uavs, runtime):
            state = uav.data.root_state_w
            quat_wxyz = state[:, 3:7]
            v_world = state[:, 7:10]
            w_world = state[:, 10:13]
            body_com_b = uav.data.body_com_pos_b[:, data["body_id"]]

            v_body = math_utils.quat_apply_inverse(quat_wxyz, v_world)
            w_body = math_utils.quat_apply_inverse(quat_wxyz, w_world)
            roll, pitch, yaw = _euler_from_quat_wxyz(quat_wxyz)

            control_out = data["autopilot"].step(
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

            data["throttle_state"] = data["throttle_state"] + (sim_dt / thrust_tau) * (
                control_out["throttle_cmd"] - data["throttle_state"]
            )
            data["throttle_state"] = _clamp(data["throttle_state"], 0.0, 1.0)

            if args_cli.propeller_anim and data["propeller_joint_ids"]:
                data["propeller_effort_target"] = torch.where(
                    data["throttle_state"].unsqueeze(-1) >= float(args_cli.propeller_anim_threshold),
                    data["throttle_state"].unsqueeze(-1)
                    * float(args_cli.propeller_anim_effort)
                    * float(args_cli.propeller_anim_sign),
                    torch.zeros_like(data["propeller_effort_target"]),
                )
                uav.set_joint_effort_target(data["propeller_effort_target"], joint_ids=data["propeller_joint_ids"])
            else:
                data["propeller_effort_target"].zero_()

            thrust_force = torch.stack(
                [
                    data["throttle_state"] * float(max_thrust),
                    torch.zeros_like(data["throttle_state"]),
                    torch.zeros_like(data["throttle_state"]),
                ],
                dim=-1,
            )
            forces_flu, moments_flu = compute_aero_forces_and_moments(
                v_body=v_body,
                w_body=w_body,
                aileron=control_out["aileron_cmd"],
                elevator=control_out["elevator_cmd"],
                rudder=control_out["rudder_cmd"],
                cfg=aero_cfg,
            )
            total_forces = _clamp(thrust_force + forces_flu, -MAX_FORCE, MAX_FORCE)
            total_torques = _clamp(moments_flu, -MAX_TORQUE, MAX_TORQUE)

            uav.instantaneous_wrench_composer.add_forces_and_torques(
                forces=total_forces.unsqueeze(1),
                positions=body_com_b.unsqueeze(1),
                body_ids=data["body_ids_t"],
            )
            uav.instantaneous_wrench_composer.add_forces_and_torques(
                torques=total_torques.unsqueeze(1),
                body_ids=data["body_ids_t"],
            )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if follow_cam is not None:
            if follow_cam_ui is not None:
                (
                    next_uav_id,
                    next_env_id,
                    distance,
                    height,
                    lookahead,
                    smooth_tau,
                    reset_requested,
                ) = follow_cam_ui.get_inputs()
                reset_requested = reset_requested or next_uav_id != follow_uav_id
                follow_uav_id = next_uav_id
                follow_cam.update_config(
                    env_id=next_env_id,
                    distance=distance,
                    height=height,
                    lookahead=lookahead,
                    smooth_tau=smooth_tau,
                    reset_smoothing=reset_requested,
                )
            follow_cam.step(uavs[follow_uav_id].data.root_state_w)


def main():
    if args_cli.num_uavs < 1:
        raise ValueError("--num_uavs must be at least 1.")

    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-120.0, -120.0, 80.0], [0.0, 0.0, args_cli.start_alt])

    scene_cfg = _make_scene_cfg(
        num_envs=int(args_cli.num_envs),
        env_spacing=float(args_cli.env_spacing),
        num_uavs=int(args_cli.num_uavs),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    uavs = [scene[f"uav_{uav_id}"] for uav_id in range(int(args_cli.num_uavs))]
    _init_formation_state(
        uavs=uavs,
        scene=scene,
        altitude=float(args_cli.start_alt),
        speed=float(args_cli.start_speed),
        roll_deg=float(args_cli.start_roll_deg),
        pitch_deg=float(args_cli.start_pitch_deg),
        yaw_deg=float(args_cli.start_yaw_deg),
        longitudinal_spacing=float(args_cli.longitudinal_spacing),
        lateral_spacing=float(args_cli.lateral_spacing),
    )
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    print("[INFO]: Multi-UAV inverted-V init complete. Running common fixed-wing autopilot scene.")
    run_simulator(sim, scene, uavs)


if __name__ == "__main__":
    main()
    simulation_app.close()
