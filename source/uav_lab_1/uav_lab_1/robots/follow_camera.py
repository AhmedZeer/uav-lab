from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils


@dataclass
class FollowCameraConfig:
    env_id: int = 0
    distance: float = 10.0
    height: float = 3.0
    lookahead: float = 6.0
    smooth_tau: float = 0.35
    min_speed_xy: float = 0.8
    target_height_offset: float = 0.5


class SmoothedFollowCamera:
    def __init__(
        self,
        sim,
        sim_dt: float,
        device: torch.device,
        num_envs: int,
        cfg: FollowCameraConfig | None = None,
    ):
        self._sim = sim
        self._device = device
        self._sim_dt = float(sim_dt)
        self._num_envs = int(num_envs)
        self.cfg = cfg if cfg is not None else FollowCameraConfig()

        self.env_id = self._clamp_env_id(self.cfg.env_id)
        self._update_alpha()

        self._heading = None
        self._eye = None
        self._target = None

        self._forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=self._device)
        self._eye_height_offset = torch.tensor([0.0, 0.0, float(self.cfg.height)], device=self._device)
        self._target_height_offset = torch.tensor([0.0, 0.0, float(self.cfg.target_height_offset)], device=self._device)

    def _clamp_env_id(self, env_id: int) -> int:
        if self._num_envs <= 0:
            return 0
        return int(max(0, min(self._num_envs - 1, int(env_id))))

    def _update_alpha(self) -> None:
        tau = max(0.0, float(self.cfg.smooth_tau))
        self._alpha = 1.0 if tau <= 0.0 else self._sim_dt / (tau + self._sim_dt)

    def update_config(
        self,
        *,
        env_id: int | None = None,
        distance: float | None = None,
        height: float | None = None,
        lookahead: float | None = None,
        smooth_tau: float | None = None,
        target_height_offset: float | None = None,
        reset_smoothing: bool = False,
    ) -> None:
        """Update camera parameters while the simulation is running."""
        if env_id is not None:
            next_env_id = self._clamp_env_id(env_id)
            reset_smoothing = reset_smoothing or next_env_id != self.env_id
            self.env_id = next_env_id
            self.cfg.env_id = next_env_id
        if distance is not None:
            self.cfg.distance = float(distance)
        if height is not None:
            self.cfg.height = float(height)
            self._eye_height_offset = torch.tensor([0.0, 0.0, float(height)], device=self._device)
        if lookahead is not None:
            self.cfg.lookahead = float(lookahead)
        if smooth_tau is not None:
            self.cfg.smooth_tau = float(smooth_tau)
            self._update_alpha()
        if target_height_offset is not None:
            self.cfg.target_height_offset = float(target_height_offset)
            self._target_height_offset = torch.tensor(
                [0.0, 0.0, float(target_height_offset)], device=self._device
            )
        if reset_smoothing:
            self.reset_smoothing()

    def reset_smoothing(self) -> None:
        self._heading = None
        self._eye = None
        self._target = None

    @staticmethod
    def _normalize(vec: torch.Tensor) -> torch.Tensor:
        return vec / torch.clamp(torch.linalg.norm(vec), min=1.0e-6)

    def _compute_heading(self, quat_wxyz: torch.Tensor, vel_world: torch.Tensor) -> torch.Tensor:
        vel_xy = vel_world[0:2]
        speed_xy = torch.linalg.norm(vel_xy).item()
        if speed_xy > float(self.cfg.min_speed_xy):
            heading_xy = self._normalize(vel_xy)
            return torch.stack([heading_xy[0], heading_xy[1], heading_xy.new_tensor(0.0)])

        forward_world = math_utils.quat_apply(quat_wxyz.unsqueeze(0), self._forward_local)[0]
        heading = forward_world.clone()
        heading[2] = 0.0
        return self._normalize(heading)

    def step(self, root_state_w: torch.Tensor):
        pos = root_state_w[self.env_id, 0:3]
        quat = root_state_w[self.env_id, 3:7]
        vel = root_state_w[self.env_id, 7:10]

        heading = self._compute_heading(quat, vel)
        if self._heading is None:
            self._heading = heading.clone()
        else:
            self._heading = self._normalize(self._heading + self._alpha * (heading - self._heading))

        desired_eye = pos - self._heading * float(self.cfg.distance) + self._eye_height_offset
        desired_target = pos + self._heading * float(self.cfg.lookahead) + self._target_height_offset

        if self._eye is None:
            self._eye = desired_eye.clone()
            self._target = desired_target.clone()
        else:
            self._eye = self._eye + self._alpha * (desired_eye - self._eye)
            self._target = self._target + self._alpha * (desired_target - self._target)

        self._sim.set_camera_view(
            self._eye.detach().cpu().tolist(),
            self._target.detach().cpu().tolist(),
        )
