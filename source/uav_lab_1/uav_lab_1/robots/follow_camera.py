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
        self.cfg = cfg if cfg is not None else FollowCameraConfig()

        if num_envs <= 0:
            self.env_id = 0
        else:
            self.env_id = int(max(0, min(num_envs - 1, int(self.cfg.env_id))))

        tau = max(0.0, float(self.cfg.smooth_tau))
        self._alpha = 1.0 if tau <= 0.0 else float(sim_dt) / (tau + float(sim_dt))

        self._heading = None
        self._eye = None
        self._target = None

        self._forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=self._device)
        self._eye_height_offset = torch.tensor([0.0, 0.0, float(self.cfg.height)], device=self._device)
        self._target_height_offset = torch.tensor([0.0, 0.0, float(self.cfg.target_height_offset)], device=self._device)

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
