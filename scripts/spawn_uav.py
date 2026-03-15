
import argparse

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
parser.add_argument("--ui", action="store_true", help="Enable UI controls for thrust/roll/pitch/yaw.")
parser.add_argument("--vis_markers", action="store_true", help="Show frame/force visualization markers.")
parser.add_argument("--force_vis_scale", type=float, default=0.02, help="Scale for force arrow length.")
parser.add_argument(
    "--thrust_pos",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help="Thrust application point in body frame (x y z).",
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
from uav_lab_1.scenes import BasicFixedWing1SceneCfg

MAX_FORCE = 1000.0
MAX_TORQUE = 500.0

PROP_CFG = PropConfig()
CONTROL_SIGNS = ControlSigns()
DRAG_CFG = DragConfig()
AERO_CFG = AeroConfig()


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
            thrust_positions[:, 0, :] = torch.tensor(args_cli.thrust_pos, device=device)
            com_positions = torch.zeros((num_envs, len(body_ids), 3), device=device)
            com_positions[:, 0, :] = uav.data.body_com_pos_b[:, 0]

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
