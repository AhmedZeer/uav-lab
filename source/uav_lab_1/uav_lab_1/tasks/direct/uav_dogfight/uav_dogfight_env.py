from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils import math as math_utils

from uav_lab_1.controllers import FixedWingAutopilot, FixedWingAutopilotConfig
from uav_lab_1.robots.aero import AeroConfig, compute_aero_forces_and_moments
from uav_lab_1.robots.follow_camera import FollowCameraConfig, SmoothedFollowCamera

from .uav_dogfight_env_cfg import UavDogfightEnvCfg


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def _sample_uniform(shape: tuple[int, ...], lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(lo, hi)


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


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


def _euler_from_quat_wxyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class UavDogfightEnv(DirectMARLEnv):
    cfg: UavDogfightEnvCfg

    def __init__(self, cfg: UavDogfightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._agents = ("uav_0", "uav_1")
        self._robots = {"uav_0": self.uav_0, "uav_1": self.uav_1}
        self._opponents = {"uav_0": "uav_1", "uav_1": "uav_0"}

        self._actions = {
            agent: torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32) for agent in self._agents
        }
        self._prev_actions = {
            agent: torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32) for agent in self._agents
        }
        self._throttle = {
            agent: torch.full((self.num_envs,), self.cfg.min_throttle, device=self.device, dtype=torch.float32)
            for agent in self._agents
        }
        self._prev_range = {
            agent: torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32) for agent in self._agents
        }
        self._last_metrics = {}

        self._aero_cfg = AeroConfig()
        autopilot_cfg = FixedWingAutopilotConfig(dt=self.physics_dt)
        self._autopilots = {
            agent: FixedWingAutopilot(autopilot_cfg, self.num_envs, self.device) for agent in self._agents
        }

        self._body_ids = {}
        self._body_ids_t = {}
        for agent, robot in self._robots.items():
            body_ids, _ = robot.find_bodies("body")
            body_id = int(body_ids[0]) if len(body_ids) > 0 else 0
            self._body_ids[agent] = body_id
            self._body_ids_t[agent] = torch.tensor([body_id], device=self.device, dtype=torch.int32)

        self._debug_follow_camera = None
        if self.cfg.debug_follow_camera and self.sim.has_gui():
            self._debug_follow_camera = SmoothedFollowCamera(
                sim=self.sim,
                sim_dt=self.physics_dt,
                device=self.device,
                num_envs=self.num_envs,
                cfg=FollowCameraConfig(
                    env_id=int(self.cfg.debug_follow_camera_env_id),
                    distance=float(self.cfg.debug_follow_camera_distance),
                    height=float(self.cfg.debug_follow_camera_height),
                    lookahead=float(self.cfg.debug_follow_camera_lookahead),
                    smooth_tau=float(self.cfg.debug_follow_camera_smooth_tau),
                ),
            )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in [
                "angle",
                "range",
                "lock_ready",
                "closing",
                "tailed",
                "too_close",
                "bounds",
                "bad_speed",
                "action_jerk",
            ]
        }

    def _setup_scene(self):
        self.uav_0 = Articulation(self.cfg.uav_0_cfg)
        self.uav_1 = Articulation(self.cfg.uav_1_cfg)

        ground_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.ground_usd_path)
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["uav_0"] = self.uav_0
        self.scene.articulations["uav_1"] = self.uav_1

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self._agents:
            self._prev_actions[agent] = self._actions[agent].clone()
            self._actions[agent] = actions[agent].clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        for agent in self._agents:
            robot = self._robots[agent]
            state = robot.data.root_state_w
            quat_wxyz = state[:, 3:7]
            v_world = state[:, 7:10]
            w_world = state[:, 10:13]
            v_body = math_utils.quat_apply_inverse(quat_wxyz, v_world)
            w_body = math_utils.quat_apply_inverse(quat_wxyz, w_world)
            roll, pitch, yaw = _euler_from_quat_wxyz(quat_wxyz)

            action = self._actions[agent]
            throttle_sp = self.cfg.min_throttle + 0.5 * (action[:, 0] + 1.0) * (
                self.cfg.max_throttle - self.cfg.min_throttle
            )
            control_out = self._autopilots[agent].step(
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                v_body=v_body,
                w_body=w_body,
                thrust_setpoint=throttle_sp,
                roll_setpoint_norm=action[:, 1],
                pitch_setpoint_norm=action[:, 2],
                yaw_setpoint_norm=torch.zeros_like(action[:, 2]),
            )

            alpha = min(self.physics_dt / max(float(self.cfg.thrust_tau), 1.0e-4), 1.0)
            self._throttle[agent] = self._throttle[agent] + alpha * (
                control_out["throttle_cmd"] - self._throttle[agent]
            )
            self._throttle[agent] = _clamp(self._throttle[agent], 0.0, 1.0)

            thrust_force = torch.stack(
                [
                    self._throttle[agent] * float(self.cfg.max_thrust),
                    torch.zeros_like(self._throttle[agent]),
                    torch.zeros_like(self._throttle[agent]),
                ],
                dim=-1,
            )
            forces_flu, moments_flu = compute_aero_forces_and_moments(
                v_body=v_body,
                w_body=w_body,
                aileron=control_out["aileron_cmd"],
                elevator=control_out["elevator_cmd"],
                rudder=control_out["rudder_cmd"],
                cfg=self._aero_cfg,
            )
            total_forces = _clamp(thrust_force + forces_flu, -self.cfg.max_force, self.cfg.max_force)
            total_torques = _clamp(moments_flu, -self.cfg.max_torque, self.cfg.max_torque)

            body_id = self._body_ids[agent]
            com_positions = robot.data.body_com_pos_b[:, body_id].unsqueeze(1)
            forces = total_forces.unsqueeze(1)
            torques = total_torques.unsqueeze(1)
            robot.instantaneous_wrench_composer.add_forces_and_torques(
                forces=forces,
                positions=com_positions,
                body_ids=self._body_ids_t[agent],
            )
            robot.instantaneous_wrench_composer.add_forces_and_torques(
                torques=torques,
                body_ids=self._body_ids_t[agent],
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {}
        self._last_metrics = {}
        for agent in self._agents:
            metrics = self._compute_pair_metrics(agent)
            robot = self._robots[agent]
            obs = torch.cat(
                [
                    robot.data.root_lin_vel_b / self.cfg.rel_vel_scale,
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    self._throttle[agent].unsqueeze(-1),
                    metrics["rel_pos_b"] / self.cfg.rel_pos_scale,
                    metrics["rel_vel_b"] / self.cfg.rel_vel_scale,
                    metrics["target_forward_b"],
                    (metrics["range"] / self.cfg.range_scale).unsqueeze(-1),
                    metrics["bearing"].unsqueeze(-1),
                    metrics["elevation"].unsqueeze(-1),
                    self._prev_actions[agent],
                ],
                dim=-1,
            )
            observations[agent] = obs
            self._last_metrics[agent] = metrics
        self._update_debug_camera()
        return observations

    def _get_states(self) -> torch.Tensor:
        obs = self._get_observations()
        return torch.cat([obs[agent] for agent in self._agents], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        current_metrics = {agent: self._compute_pair_metrics(agent) for agent in self._agents}
        rewards = {}
        for agent in self._agents:
            metrics = current_metrics[agent]

            angle_error = torch.sqrt(metrics["bearing"] ** 2 + metrics["elevation"] ** 2)
            in_front = metrics["rel_pos_b"][:, 0] > 0.0
            angle_reward = torch.exp(-angle_error / 0.35) * in_front.float()

            range_error = torch.abs(metrics["range"] - self.cfg.desired_range)
            range_reward = torch.exp(-range_error / max(self.cfg.desired_range, 1.0))

            closing = torch.clamp(self._prev_range[agent] - metrics["range"], -10.0, 10.0)
            first_step = self._prev_range[agent] <= 1.0e-3
            closing = torch.where(first_step, torch.zeros_like(closing), closing)

            lock_ready = (
                (metrics["range"] > self.cfg.min_range)
                & (metrics["range"] < self.cfg.lock_range)
                & (torch.abs(metrics["bearing"]) < self.cfg.lock_bearing_rad)
                & (torch.abs(metrics["elevation"]) < self.cfg.lock_elevation_rad)
                & in_front
            )

            opponent = self._opponents[agent]
            opponent_metrics = current_metrics[opponent]
            tailed = (
                (opponent_metrics["range"] > self.cfg.min_range)
                & (opponent_metrics["range"] < self.cfg.lock_range)
                & (torch.abs(opponent_metrics["bearing"]) < self.cfg.lock_bearing_rad)
                & (torch.abs(opponent_metrics["elevation"]) < self.cfg.lock_elevation_rad)
                & (opponent_metrics["rel_pos_b"][:, 0] > 0.0)
            )

            too_close = metrics["range"] < self.cfg.min_range
            out_of_bounds = self._out_of_bounds(self._robots[agent])
            bad_speed = self._bad_speed(self._robots[agent])
            action_jerk = torch.sum(torch.square(self._actions[agent] - self._prev_actions[agent]), dim=-1)

            components = {
                "angle": self.cfg.rew_angle * angle_reward,
                "range": self.cfg.rew_range * range_reward,
                "lock_ready": self.cfg.rew_lock_ready * lock_ready.float(),
                "closing": self.cfg.rew_closing * closing,
                "tailed": self.cfg.rew_tailed * tailed.float(),
                "too_close": self.cfg.rew_too_close * too_close.float(),
                "bounds": self.cfg.rew_bounds * out_of_bounds.float(),
                "bad_speed": self.cfg.rew_bad_speed * bad_speed.float(),
                "action_jerk": self.cfg.rew_action_jerk * action_jerk,
            }
            reward = torch.full((self.num_envs,), self.cfg.rew_alive, device=self.device, dtype=torch.float32)
            for key, value in components.items():
                reward = reward + value
                self._episode_sums[key] += value
            rewards[agent] = reward
            self._prev_range[agent] = metrics["range"].clone()

        if "log" not in self.extras:
            self.extras["log"] = {}
        for key, value in self._episode_sums.items():
            self.extras["log"][f"Reward/{key}"] = value.mean()
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated_any = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        for agent in self._agents:
            terminated_any |= self._out_of_bounds(self._robots[agent])
            terminated_any |= self._bad_speed(self._robots[agent])

        terminated = {agent: terminated_any for agent in self._agents}
        time_outs = {agent: time_out for agent in self._agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.uav_0._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._reset_pair_states(env_ids)
        for agent in self._agents:
            self._actions[agent][env_ids] = 0.0
            self._prev_actions[agent][env_ids] = 0.0
            self._throttle[agent][env_ids] = self.cfg.min_throttle
            self._prev_range[agent][env_ids] = 0.0
            self._autopilots[agent].reset(env_ids)

        if "log" not in self.extras:
            self.extras["log"] = {}
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0

    def _compute_pair_metrics(self, agent: str) -> dict[str, torch.Tensor]:
        robot = self._robots[agent]
        target = self._robots[self._opponents[agent]]
        own_pos = robot.data.root_pos_w
        own_quat = robot.data.root_quat_w
        target_pos = target.data.root_pos_w
        target_quat = target.data.root_quat_w

        rel_pos_w = target_pos - own_pos
        rel_vel_w = target.data.root_lin_vel_w - robot.data.root_lin_vel_w
        rel_pos_b = math_utils.quat_apply_inverse(own_quat, rel_pos_w)
        rel_vel_b = math_utils.quat_apply_inverse(own_quat, rel_vel_w)
        target_forward_w = math_utils.quat_apply(
            target_quat,
            torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).expand(self.num_envs, -1),
        )
        target_forward_b = math_utils.quat_apply_inverse(own_quat, target_forward_w)
        distance = torch.linalg.norm(rel_pos_w, dim=-1)
        safe_x = torch.clamp(rel_pos_b[:, 0], min=1.0e-3)
        bearing = torch.atan2(rel_pos_b[:, 1], safe_x)
        elevation = torch.atan2(rel_pos_b[:, 2], safe_x)

        return {
            "rel_pos_b": rel_pos_b,
            "rel_vel_b": rel_vel_b,
            "target_forward_b": target_forward_b,
            "range": distance,
            "bearing": bearing,
            "elevation": elevation,
        }

    def _out_of_bounds(self, robot: Articulation) -> torch.Tensor:
        pos = robot.data.root_pos_w
        rel_xy = pos[:, :2] - self.scene.env_origins[:, :2]
        horizontal = torch.linalg.norm(rel_xy, dim=-1) > self.cfg.arena_radius
        altitude = (pos[:, 2] < self.cfg.min_altitude) | (pos[:, 2] > self.cfg.max_altitude)
        return horizontal | altitude

    def _bad_speed(self, robot: Articulation) -> torch.Tensor:
        speed = torch.linalg.norm(robot.data.root_lin_vel_b, dim=-1)
        return (speed < self.cfg.min_speed) | (speed > self.cfg.max_speed)

    def _reset_pair_states(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        angle = _sample_uniform((n,), -math.pi, math.pi, self.device)
        separation = _sample_uniform((n,), self.cfg.reset_separation_min, self.cfg.reset_separation_max, self.device)
        offset = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1) * (
            0.5 * separation
        ).unsqueeze(-1)
        altitude = self.cfg.reset_altitude + _sample_uniform(
            (n,), -self.cfg.reset_altitude_noise, self.cfg.reset_altitude_noise, self.device
        )

        yaw_noise = math.radians(self.cfg.reset_yaw_noise_deg)
        yaw_0 = _wrap_angle(angle + _sample_uniform((n,), -yaw_noise, yaw_noise, self.device))
        yaw_1 = _wrap_angle(angle + math.pi + _sample_uniform((n,), -yaw_noise, yaw_noise, self.device))
        speed_0 = _sample_uniform((n,), self.cfg.reset_speed_min, self.cfg.reset_speed_max, self.device)
        speed_1 = _sample_uniform((n,), self.cfg.reset_speed_min, self.cfg.reset_speed_max, self.device)

        self._write_aircraft_state(
            self.uav_0,
            env_ids,
            origins - offset + torch.stack([torch.zeros_like(altitude), torch.zeros_like(altitude), altitude], dim=-1),
            yaw_0,
            speed_0,
        )
        self._write_aircraft_state(
            self.uav_1,
            env_ids,
            origins + offset + torch.stack([torch.zeros_like(altitude), torch.zeros_like(altitude), altitude], dim=-1),
            yaw_1,
            speed_1,
        )

    def _write_aircraft_state(
        self,
        robot: Articulation,
        env_ids: torch.Tensor,
        pos_w: torch.Tensor,
        yaw: torch.Tensor,
        speed: torch.Tensor,
    ) -> None:
        n = len(env_ids)
        joint_pos = robot.data.default_joint_pos[env_ids]
        joint_vel = robot.data.default_joint_vel[env_ids]

        root_pose = robot.data.default_root_state[env_ids, :7].clone()
        root_vel = robot.data.default_root_state[env_ids, 7:].clone()
        root_pose[:, :3] = pos_w
        quat_wxyz = _quat_from_rpy_wxyz(
            torch.zeros((n,), device=self.device),
            torch.zeros((n,), device=self.device),
            yaw,
        )
        root_pose[:, 3:7] = quat_wxyz

        lin_vel_body = torch.stack(
            [speed, torch.zeros_like(speed), torch.zeros_like(speed)],
            dim=-1,
        )
        root_vel[:, 0:3] = math_utils.quat_apply(quat_wxyz, lin_vel_body)
        root_vel[:, 3:6] = 0.0

        robot.write_root_pose_to_sim(root_pose, env_ids)
        robot.write_root_velocity_to_sim(root_vel, env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _update_debug_camera(self) -> None:
        if self._debug_follow_camera is not None:
            self._debug_follow_camera.step(self.uav_0.data.root_state_w)
