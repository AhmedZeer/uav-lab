from __future__ import annotations

from dataclasses import dataclass

import torch


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


@dataclass
class FixedWingAutopilotConfig:
    dt: float
    roll_setpoint_limit_deg: float = 35.0
    pitch_setpoint_limit_deg: float = 20.0
    heading_offset_limit_deg: float = 120.0
    heading_bank_limit_deg: float = 25.0
    roll_kp: float = 2.6
    roll_ki: float = 0.35
    roll_kd: float = 0.0
    pitch_kp: float = 3.4
    pitch_kd: float = 0.00
    heading_kp: float = 0.0
    heading_ki: float = 0.0
    sideslip_kp: float = 2.4
    sideslip_ki: float = 0.35
    yaw_damper_kd: float = 0.2
    roll_integrator_limit: float = 0.75
    heading_integrator_limit_deg: float = 30.0
    sideslip_integrator_limit_deg: float = 20.0
    min_airspeed_for_sideslip_mps: float = 3.0


class FixedWingAutopilot:
    """Successive-loop-closure-inspired fixed-wing attitude controller.

    The controller keeps the current heading as its reference when the yaw
    setpoint input is zero, so the UI can command heading offsets rather than
    requiring world-frame absolute headings.
    """

    def __init__(self, cfg: FixedWingAutopilotConfig, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device

        self.roll_limit = torch.deg2rad(torch.tensor(cfg.roll_setpoint_limit_deg, device=device, dtype=torch.float32))
        self.pitch_limit = torch.deg2rad(
            torch.tensor(cfg.pitch_setpoint_limit_deg, device=device, dtype=torch.float32)
        )
        self.heading_offset_limit = torch.deg2rad(
            torch.tensor(cfg.heading_offset_limit_deg, device=device, dtype=torch.float32)
        )
        self.heading_bank_limit = torch.deg2rad(
            torch.tensor(cfg.heading_bank_limit_deg, device=device, dtype=torch.float32)
        )
        self.heading_integrator_limit = torch.deg2rad(
            torch.tensor(cfg.heading_integrator_limit_deg, device=device, dtype=torch.float32)
        )
        self.sideslip_integrator_limit = torch.deg2rad(
            torch.tensor(cfg.sideslip_integrator_limit_deg, device=device, dtype=torch.float32)
        )

        self.heading_reference = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
        self.heading_reference_initialized = torch.zeros((self.num_envs,), device=device, dtype=torch.bool)

        self.roll_integrator = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
        self.roll_error_d1 = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
        self.heading_integrator = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
        self.heading_error_d1 = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
        self.sideslip_integrator = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)
        self.sideslip_error_d1 = torch.zeros((self.num_envs,), device=device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None, yaw: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if yaw is not None:
            self.heading_reference[env_ids] = yaw[env_ids]
            self.heading_reference_initialized[env_ids] = True
        else:
            self.heading_reference[env_ids] = 0.0
            self.heading_reference_initialized[env_ids] = False
        self.roll_integrator[env_ids] = 0.0
        self.roll_error_d1[env_ids] = 0.0
        self.heading_integrator[env_ids] = 0.0
        self.heading_error_d1[env_ids] = 0.0
        self.sideslip_integrator[env_ids] = 0.0
        self.sideslip_error_d1[env_ids] = 0.0

    def _update_heading_reference(self, yaw: torch.Tensor):
        mask = ~self.heading_reference_initialized
        if torch.any(mask):
            self.heading_reference[mask] = yaw[mask]
            self.heading_reference_initialized[mask] = True

    def _integrate_trapezoid(
        self,
        integrator: torch.Tensor,
        error: torch.Tensor,
        error_d1: torch.Tensor,
        limit: float | torch.Tensor,
    ) -> torch.Tensor:
        next_integrator = integrator + (0.5 * self.cfg.dt * (error + error_d1))
        if isinstance(limit, torch.Tensor):
            return torch.minimum(torch.maximum(next_integrator, -limit), limit)
        return _clamp(next_integrator, -float(limit), float(limit))

    def step(
        self,
        *,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        yaw: torch.Tensor,
        v_body: torch.Tensor,
        w_body: torch.Tensor,
        thrust_setpoint: torch.Tensor,
        roll_setpoint_norm: torch.Tensor,
        pitch_setpoint_norm: torch.Tensor,
        yaw_setpoint_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._update_heading_reference(yaw)

        roll_setpoint = _clamp(roll_setpoint_norm, -1.0, 1.0) * self.roll_limit
        pitch_setpoint = _clamp(pitch_setpoint_norm, -1.0, 1.0) * self.pitch_limit
        heading_setpoint = _wrap_angle(
            self.heading_reference + (_clamp(yaw_setpoint_norm, -1.0, 1.0) * self.heading_offset_limit)
        )

        heading_error = _wrap_angle(heading_setpoint - yaw)
        self.heading_integrator = self._integrate_trapezoid(
            self.heading_integrator,
            heading_error,
            self.heading_error_d1,
            self.heading_integrator_limit,
        )
        heading_bank_cmd = (
            self.cfg.heading_kp * heading_error
            + self.cfg.heading_ki * self.heading_integrator
        )
        heading_bank_cmd = _clamp(
            heading_bank_cmd,
            -float(self.heading_bank_limit.item()),
            float(self.heading_bank_limit.item()),
        )
        self.heading_error_d1 = heading_error

        roll_command = _clamp(
            roll_setpoint + heading_bank_cmd,
            -float(self.roll_limit.item()),
            float(self.roll_limit.item()),
        )
        roll_error = roll_command - roll
        self.roll_integrator = self._integrate_trapezoid(
            self.roll_integrator,
            roll_error,
            self.roll_error_d1,
            self.cfg.roll_integrator_limit,
        )
        roll_rate = w_body[:, 0]
        aileron_cmd = (
            self.cfg.roll_kp * roll_error
            + self.cfg.roll_ki * self.roll_integrator
            - self.cfg.roll_kd * roll_rate
        )
        aileron_cmd = _clamp(aileron_cmd, -1.0, 1.0)
        self.roll_error_d1 = roll_error

        pitch_error = pitch_setpoint - pitch
        pitch_rate = w_body[:, 1]
        elevator_cmd = self.cfg.pitch_kp * pitch_error - self.cfg.pitch_kd * pitch_rate
        elevator_cmd = _clamp(elevator_cmd, -1.0, 1.0)

        speed = torch.linalg.norm(v_body, dim=-1)
        safe_speed = torch.clamp(speed, min=self.cfg.min_airspeed_for_sideslip_mps)
        sideslip = torch.asin(torch.clamp(-v_body[:, 1] / safe_speed, min=-1.0, max=1.0))
        low_speed = speed < self.cfg.min_airspeed_for_sideslip_mps
        sideslip = torch.where(low_speed, torch.zeros_like(sideslip), sideslip)
        sideslip_error = -sideslip
        self.sideslip_integrator = self._integrate_trapezoid(
            self.sideslip_integrator,
            sideslip_error,
            self.sideslip_error_d1,
            self.sideslip_integrator_limit,
        )
        yaw_rate = w_body[:, 2]
        rudder_cmd = (
            self.cfg.sideslip_kp * sideslip_error
            + self.cfg.sideslip_ki * self.sideslip_integrator
            - self.cfg.yaw_damper_kd * yaw_rate
        )
        rudder_cmd = _clamp(rudder_cmd, -1.0, 1.0)
        self.sideslip_error_d1 = sideslip_error

        throttle_cmd = _clamp(thrust_setpoint, 0.0, 1.0)
        axis_command = torch.stack([aileron_cmd, elevator_cmd, rudder_cmd], dim=-1)

        return {
            "throttle_cmd": throttle_cmd,
            "axis_command": axis_command,
            "aileron_cmd": aileron_cmd,
            "elevator_cmd": elevator_cmd,
            "rudder_cmd": rudder_cmd,
            "roll_setpoint": roll_setpoint,
            "pitch_setpoint": pitch_setpoint,
            "heading_setpoint": heading_setpoint,
            "roll_command": roll_command,
            "heading_bank_cmd": heading_bank_cmd,
            "sideslip": sideslip,
            "heading_error": heading_error,
            "pitch_error": pitch_error,
            "roll_error" : roll_error,
            "sideslip_error": sideslip_error,
        }
