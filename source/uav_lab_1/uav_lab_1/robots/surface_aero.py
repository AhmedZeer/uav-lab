from dataclasses import dataclass

import torch


@dataclass
class LiftingSurfaceConfig:
    name: str
    position_b: tuple[float, float, float]
    lift_unit_b: tuple[float, float, float]
    forward_unit_b: tuple[float, float, float]
    cl_alpha_2d: float
    chord: float
    span: float
    flap_to_chord: float
    eta: float
    alpha_0_deg: float
    alpha_stall_p_deg: float
    alpha_stall_n_deg: float
    cd_0: float
    deflection_limit_deg: float
    tau: float


def default_fixedwing_surface_configs() -> list[LiftingSurfaceConfig]:
    """Measured fixed-wing surface layout for the current USD asset.

    Notes:
        - These dimensions follow the measured aircraft in the USD stage rather
          than the original paper/PyFlyt reference airframe.
        - Full wingspan is 4.44 m, wing chord is 0.48 m, and the outboard
          aileron span is 1.31 m per side.
        - The horizontal tail is measured at about 1.06 m span and 0.23 m
          chord.
        - The vertical tail is measured at about 0.49 m height and 0.32 m
          chord.
        - The tail reference point is placed about 1.41 m aft along body -X.
        - `alpha_0_deg` is neutralized to 0.0 deg because the old paper trim
          was injecting large zero-command lift on this asset.
    """
    wing_chord = 0.48
    full_wing_span = 4.44
    aileron_span = 1.31
    center_wing_span = full_wing_span - (2.0 * aileron_span)
    aileron_center_y = (0.5 * center_wing_span) + (0.5 * aileron_span)

    tail_arm_x = -1.41
    h_tail_full_span = 1.06
    h_tail_chord = 0.23
    v_tail_chord = 0.32
    v_tail_height = 0.49
    v_tail_center_z = 0.5 * v_tail_height

    return [
        LiftingSurfaceConfig(
            name="left_aileron",
            position_b=(0.0, aileron_center_y, 0.0),
            lift_unit_b=(0.0, 0.0, 1.0),
            forward_unit_b=(1.0, 0.0, 0.0),
            # cl_alpha_2d=6.283,
            # cl_alpha_2d=2.0,
            cl_alpha_2d=1.5,
            chord=wing_chord,
            span=aileron_span,
            flap_to_chord=0.3,
            eta=0.65,
            alpha_0_deg=0.0,
            alpha_stall_p_deg=14.0,
            alpha_stall_n_deg=-9.0,
            # cd_0=0.01,
            # cd_0=0.03,
            cd_0=0.03,
            deflection_limit_deg=30.0,
            tau=0.05,
        ),
        LiftingSurfaceConfig(
            name="right_aileron",
            position_b=(0.0, -aileron_center_y, 0.0),
            lift_unit_b=(0.0, 0.0, 1.0),
            forward_unit_b=(1.0, 0.0, 0.0),
            # cl_alpha_2d=6.283,
            # cl_alpha_2d=2.0,
            cl_alpha_2d=1.5,
            chord=wing_chord,
            span=aileron_span,
            flap_to_chord=0.3,
            eta=0.65,
            alpha_0_deg=0.0,
            alpha_stall_p_deg=14.0,
            alpha_stall_n_deg=-9.0,
            # cd_0=0.01,
            # cd_0=0.03,
            cd_0=0.03,
            deflection_limit_deg=30.0,
            tau=0.05,
        ),
        LiftingSurfaceConfig(
            name="h_tail",
            # position_b=(tail_arm_x, 0.0, 0.0),
            position_b=(tail_arm_x, 0.0, 0.0),
            lift_unit_b=(0.0, 0.0, 1.0),
            forward_unit_b=(1.0, 0.0, 0.0),
            # cl_alpha_2d=6.283,
            cl_alpha_2d=1.5,
            # cl_alpha_2d=0.8,
            chord=h_tail_chord,
            span=h_tail_full_span,
            flap_to_chord=0.3,
            eta=0.65,
            alpha_0_deg=0.0,
            alpha_stall_p_deg=9.0,
            alpha_stall_n_deg=-9.0,
            # cd_0=0.01,
            cd_0=0.03,
            deflection_limit_deg=20.0,
            tau=0.05,
        ),
        LiftingSurfaceConfig(
            name="v_tail",
            position_b=(tail_arm_x, 0.0, v_tail_center_z),
            lift_unit_b=(0.0, 1.0, 0.0),
            forward_unit_b=(1.0, 0.0, 0.0),
            # cl_alpha_2d=6.283,
            cl_alpha_2d=1.5,
            chord=v_tail_chord,
            span=v_tail_height,
            flap_to_chord=0.3,
            eta=0.65,
            alpha_0_deg=0.0,
            alpha_stall_p_deg=9.0,
            alpha_stall_n_deg=-9.0,
            # cd_0=0.01,
            cd_0=0.03,
            deflection_limit_deg=20.0,
            tau=0.05,
        ),
        LiftingSurfaceConfig(
            name="main_wing",
            position_b=(0.0, 0.0, 0.0),
            lift_unit_b=(0.0, 0.0, 1.0),
            forward_unit_b=(1.0, 0.0, 0.0),
            # cl_alpha_2d=6.283,
            cl_alpha_2d=2.5,
            chord=wing_chord,
            span=center_wing_span,
            flap_to_chord=0.3,
            eta=0.65,
            alpha_0_deg=0.0,
            alpha_stall_p_deg=14.0,
            alpha_stall_n_deg=-9.0,
            # cd_0=0.01,
            cd_0=0.03,
            deflection_limit_deg=0.0,
            tau=0.05,
        ),
    ]


class SurfaceAeroModel:
    def __init__(
        self,
        surface_cfgs: list[LiftingSurfaceConfig],
        num_envs: int,
        sim_dt: float,
        device: torch.device,
        air_density: float = 1.225,
    ):
        self.surface_cfgs = surface_cfgs
        self.num_surfaces = len(surface_cfgs)
        self.num_envs = int(num_envs)
        self.sim_dt = float(sim_dt)
        self.device = device
        self.surface_names = [s.name for s in surface_cfgs]

        self.positions_b = torch.tensor([s.position_b for s in surface_cfgs], device=device, dtype=torch.float32)
        self.lift_units_b = self._normalize_rows(
            torch.tensor([s.lift_unit_b for s in surface_cfgs], device=device, dtype=torch.float32)
        )
        self.drag_units_b = self._normalize_rows(
            torch.tensor([s.forward_unit_b for s in surface_cfgs], device=device, dtype=torch.float32)
        )
        self.torque_units_b = torch.cross(self.lift_units_b, self.drag_units_b, dim=-1)
        self.torque_units_b = self._normalize_rows(self.torque_units_b)

        chord = torch.tensor([s.chord for s in surface_cfgs], device=device, dtype=torch.float32)
        span = torch.tensor([s.span for s in surface_cfgs], device=device, dtype=torch.float32)
        flap_to_chord = torch.tensor([s.flap_to_chord for s in surface_cfgs], device=device, dtype=torch.float32)
        eta = torch.tensor([s.eta for s in surface_cfgs], device=device, dtype=torch.float32)
        cl_alpha_2d = torch.tensor([s.cl_alpha_2d for s in surface_cfgs], device=device, dtype=torch.float32)
        cd_0 = torch.tensor([s.cd_0 for s in surface_cfgs], device=device, dtype=torch.float32)
        alpha_0 = torch.deg2rad(torch.tensor([s.alpha_0_deg for s in surface_cfgs], device=device, dtype=torch.float32))
        alpha_stall_p = torch.deg2rad(
            torch.tensor([s.alpha_stall_p_deg for s in surface_cfgs], device=device, dtype=torch.float32)
        )
        alpha_stall_n = torch.deg2rad(
            torch.tensor([s.alpha_stall_n_deg for s in surface_cfgs], device=device, dtype=torch.float32)
        )
        deflection_limit_deg = torch.tensor(
            [s.deflection_limit_deg for s in surface_cfgs], device=device, dtype=torch.float32
        )
        tau = torch.tensor([max(s.tau, 1.0e-3) for s in surface_cfgs], device=device, dtype=torch.float32)

        area = chord * span
        aspect = span / torch.clamp(chord, min=1.0e-6)
        cl_alpha_3d = cl_alpha_2d * (aspect / (aspect + ((2.0 * (aspect + 4.0)) / (aspect + 2.0))))
        theta_f = torch.acos(torch.clamp(2.0 * flap_to_chord - 1.0, min=-1.0, max=1.0))
        aero_tau = 1.0 - ((theta_f - torch.sin(theta_f)) / torch.pi)

        self.air_density = float(air_density)
        self.half_rho = 0.5 * self.air_density
        self.chord = chord
        self.area = area
        self.aspect = aspect
        self.flap_to_chord = flap_to_chord
        self.eta = eta
        self.cl_alpha_3d = cl_alpha_3d
        self.alpha_0_base = alpha_0
        self.alpha_stall_p_base = alpha_stall_p
        self.alpha_stall_n_base = alpha_stall_n
        self.cd_0 = cd_0
        self.deflection_limit_deg = deflection_limit_deg
        self.actuation_tau = tau
        self.aero_tau = aero_tau

        self.actuation = torch.zeros((self.num_envs, self.num_surfaces), device=device, dtype=torch.float32)
        self.alpha = torch.zeros_like(self.actuation)
        self.freestream_speed = torch.zeros_like(self.actuation)

    @staticmethod
    def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
        return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=1.0e-6)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.actuation.zero_()
            self.alpha.zero_()
            self.freestream_speed.zero_()
        else:
            self.actuation[env_ids] = 0.0
            self.alpha[env_ids] = 0.0
            self.freestream_speed[env_ids] = 0.0

    def _compute_coefficients(self, alpha: torch.Tensor, actuation: torch.Tensor):
        # Inputs are (N, S).
        deflection = torch.deg2rad(actuation * self.deflection_limit_deg.unsqueeze(0))

        delta_cl = self.cl_alpha_3d.unsqueeze(0) * self.aero_tau.unsqueeze(0) * self.eta.unsqueeze(0) * deflection
        delta_cl_max = self.flap_to_chord.unsqueeze(0) * delta_cl
        cl_max_p = self.cl_alpha_3d.unsqueeze(0) * (
            self.alpha_stall_p_base.unsqueeze(0) - self.alpha_0_base.unsqueeze(0)
        ) + delta_cl_max
        cl_max_n = self.cl_alpha_3d.unsqueeze(0) * (
            self.alpha_stall_n_base.unsqueeze(0) - self.alpha_0_base.unsqueeze(0)
        ) + delta_cl_max
        alpha_0 = self.alpha_0_base.unsqueeze(0) - (
            delta_cl / torch.clamp(self.cl_alpha_3d.unsqueeze(0), min=1.0e-6)
        )
        alpha_stall_p = alpha_0 + cl_max_p / torch.clamp(self.cl_alpha_3d.unsqueeze(0), min=1.0e-6)
        alpha_stall_n = alpha_0 + cl_max_n / torch.clamp(self.cl_alpha_3d.unsqueeze(0), min=1.0e-6)

        cl = torch.zeros_like(alpha)
        cd = torch.zeros_like(alpha)
        cm = torch.zeros_like(alpha)

        no_stall = (alpha > alpha_stall_n) & (alpha < alpha_stall_p)
        if torch.any(no_stall):
            cl_ns = self.cl_alpha_3d.unsqueeze(0) * (alpha - alpha_0)
            alpha_i_ns = cl_ns / (torch.pi * torch.clamp(self.aspect.unsqueeze(0), min=1.0e-6))
            alpha_eff_ns = alpha - alpha_0 - alpha_i_ns
            ct_ns = self.cd_0.unsqueeze(0) * torch.cos(alpha_eff_ns)
            cn_ns = (cl_ns + (ct_ns * torch.sin(alpha_eff_ns))) / torch.clamp(torch.cos(alpha_eff_ns), min=1.0e-6)
            cd_ns = (cn_ns * torch.sin(alpha_eff_ns)) + (ct_ns * torch.cos(alpha_eff_ns))
            cm_ns = -cn_ns * (0.25 - (0.175 * (1.0 - ((2.0 * alpha_eff_ns) / torch.pi))))
            cl = torch.where(no_stall, cl_ns, cl)
            cd = torch.where(no_stall, cd_ns, cd)
            cm = torch.where(no_stall, cm_ns, cm)

        stall = ~no_stall
        if torch.any(stall):
            cl_stall_p = self.cl_alpha_3d.unsqueeze(0) * (alpha_stall_p - alpha_0)
            cl_stall_n = self.cl_alpha_3d.unsqueeze(0) * (alpha_stall_n - alpha_0)
            alpha_i_at_stall_p = cl_stall_p / (torch.pi * torch.clamp(self.aspect.unsqueeze(0), min=1.0e-6))
            alpha_i_at_stall_n = cl_stall_n / (torch.pi * torch.clamp(self.aspect.unsqueeze(0), min=1.0e-6))

            pos = alpha > 0.0
            neg = ~pos

            alpha_i = torch.zeros_like(alpha)
            # alpha_i post-stall positive: interpolate to 0 at +pi/2.
            t_pos = (alpha - alpha_stall_p) / torch.clamp((torch.pi / 2.0) - alpha_stall_p, min=1.0e-6)
            t_pos = torch.clamp(t_pos, min=0.0, max=1.0)
            alpha_i_pos = alpha_i_at_stall_p * (1.0 - t_pos)
            alpha_i = torch.where(pos, alpha_i_pos, alpha_i)

            # alpha_i post-stall negative: interpolate from 0 at -pi/2 to alpha_i_at_stall_n at alpha_stall_n.
            t_neg = (alpha - (-torch.pi / 2.0)) / torch.clamp(alpha_stall_n - (-torch.pi / 2.0), min=1.0e-6)
            t_neg = torch.clamp(t_neg, min=0.0, max=1.0)
            alpha_i_neg = alpha_i_at_stall_n * t_neg
            alpha_i = torch.where(neg, alpha_i_neg, alpha_i)

            alpha_eff = alpha - alpha_0 - alpha_i
            cd_90 = (-4.26e-2) * (deflection**2) + (2.1e-1) * deflection + 1.98
            cn = cd_90 * torch.sin(alpha_eff) * (
                1.0 / (0.56 + 0.44 * torch.abs(torch.sin(alpha_eff)))
                - 0.41 * (1.0 - torch.exp(-17.0 / torch.clamp(self.aspect.unsqueeze(0), min=1.0e-6)))
            )
            ct = 0.5 * self.cd_0.unsqueeze(0) * torch.cos(alpha_eff)
            cl_st = (cn * torch.cos(alpha_eff)) - (ct * torch.sin(alpha_eff))
            cd_st = (cn * torch.sin(alpha_eff)) + (ct * torch.cos(alpha_eff))
            cm_st = -cn * (0.25 - (0.175 * (1.0 - ((2.0 * torch.abs(alpha_eff)) / torch.pi))))

            cl = torch.where(stall, cl_st, cl)
            cd = torch.where(stall, cd_st, cd)
            cm = torch.where(stall, cm_st, cm)

        return cl, cd, cm

    def step(
        self,
        v_body: torch.Tensor,
        w_body: torch.Tensor,
        cmd: torch.Tensor,
        body_com_b: torch.Tensor | None = None,
        # reference_mode: str = "config",
        reference_mode: str = "config_minus_com",
    ):
        """
        Compute aggregated aerodynamic force and torque in body frame.

        Args:
            v_body: (N, 3)
            w_body: (N, 3)
            cmd: (N, S) normalized surface command in [-1, 1]
            body_com_b: (N, 3) body COM expressed in body frame.
            reference_mode: one of {"config", "config_minus_com", "zero"}.
        """
        if cmd.shape != (self.num_envs, self.num_surfaces):
            raise ValueError(f"Expected cmd shape {(self.num_envs, self.num_surfaces)}, got {cmd.shape}")

        cmd = torch.clamp(cmd, min=-1.0, max=1.0)
        alpha = self.sim_dt / torch.clamp(self.actuation_tau.unsqueeze(0), min=1.0e-4)
        self.actuation = self.actuation + alpha * (cmd - self.actuation)
        self.actuation = torch.clamp(self.actuation, min=-1.0, max=1.0)

        raw_positions_b = self.positions_b.unsqueeze(0).expand(self.num_envs, -1, -1)
        if reference_mode == "config":
            r = raw_positions_b
        elif reference_mode == "config_minus_com":
            if body_com_b is None:
                raise ValueError("body_com_b must be provided when reference_mode='config_minus_com'")
            r = raw_positions_b - body_com_b.unsqueeze(1)
        elif reference_mode == "zero":
            r = torch.zeros_like(raw_positions_b)
        else:
            raise ValueError(
                f"Unsupported reference_mode '{reference_mode}'. Expected one of 'config', 'config_minus_com', 'zero'."
            )

        # local velocity at each surface: v + omega x r
        v = v_body.unsqueeze(1).expand(-1, self.num_surfaces, -1)
        w = w_body.unsqueeze(1).expand(-1, self.num_surfaces, -1)
        v_local = v + torch.linalg.cross(w, r, dim=-1)

        lift_speed = torch.sum(v_local * self.lift_units_b.unsqueeze(0), dim=-1)
        fwd_speed = torch.sum(v_local * self.drag_units_b.unsqueeze(0), dim=-1)
        alpha_local = torch.atan2(-lift_speed, fwd_speed)
        speed = torch.linalg.norm(v_local, dim=-1)

        self.alpha = alpha_local
        self.freestream_speed = speed

        cl, cd, cm = self._compute_coefficients(alpha_local, self.actuation)

        q = self.half_rho * speed**2
        q_area = q * self.area.unsqueeze(0)
        lift = cl * q_area
        drag = cd * q_area
        force_normal = (lift * torch.cos(alpha_local)) + (drag * torch.sin(alpha_local))
        force_parallel = (lift * torch.sin(alpha_local)) - (drag * torch.cos(alpha_local))

        force_surface = (
            force_normal.unsqueeze(-1) * self.lift_units_b.unsqueeze(0)
            + force_parallel.unsqueeze(-1) * self.drag_units_b.unsqueeze(0)
        )
        torque_surface = (
            (q_area * cm * self.chord.unsqueeze(0)).unsqueeze(-1) * self.torque_units_b.unsqueeze(0)
        )
        torque_from_force = torch.linalg.cross(r, force_surface, dim=-1)

        total_force_b = torch.sum(force_surface, dim=1)
        total_torque_b = torch.sum(torque_surface + torque_from_force, dim=1)

        return {
            "force_b": total_force_b,
            "torque_b": total_torque_b,
            "surface_force_b": force_surface,
            "surface_torque_b": torque_surface + torque_from_force,
            "surface_aero_torque_b": torque_surface,
            "surface_force_torque_b": torque_from_force,
            "surface_local_velocity_b": v_local,
            "surface_reference_pos_b": r,
            "surface_config_pos_b": raw_positions_b,
            "surface_forward_speed": fwd_speed,
            "surface_lift_speed": lift_speed,
            "alpha": self.alpha,
            "speed": self.freestream_speed,
            "actuation": self.actuation,
        }
