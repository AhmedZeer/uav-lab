import torch


class AeroConfig:
    def __init__(
        self,
        air_density: float = 1.225,
        wing_area: float = 2.36,
        wing_span: float = 4.46,
        chord: float = 0.53,
        CL_0: float = 0.3,
        CL_alpha: float = 4.0,
        CL_max: float = 1.5,
        CL_min: float = -1.1,
        CD_0: float = 0.03,
        CD_alpha: float = 0.30,
        CD_alpha2: float = 2.0,
        CY_beta: float = -0.98,
        Cm_0: float = -0.02,
        Cm_alpha: float = -0.38,
        Cl_beta: float = -0.12,
        Cl_p: float = -0.50,
        Cl_r: float = 0.15,
        Cn_beta: float = 0.25,
        Cn_p: float = -0.06,
        Cn_r: float = -0.20,
        Cm_q: float = -8.0,
        CL_elevator: float = 0.43,
        Cm_elevator: float = -1.122,
        Cl_aileron: float = 0.229,
        Cn_rudder: float = -0.032,
        CY_rudder: float = 0.870,
    ):
        self.air_density = air_density
        self.wing_area = wing_area
        self.wing_span = wing_span
        self.chord = chord
        self.CL_0 = CL_0
        self.CL_alpha = CL_alpha
        self.CL_max = CL_max
        self.CL_min = CL_min
        self.CD_0 = CD_0
        self.CD_alpha = CD_alpha
        self.CD_alpha2 = CD_alpha2
        self.CY_beta = CY_beta
        self.Cm_0 = Cm_0
        self.Cm_alpha = Cm_alpha
        self.Cl_beta = Cl_beta
        self.Cl_p = Cl_p
        self.Cl_r = Cl_r
        self.Cn_beta = Cn_beta
        self.Cn_p = Cn_p
        self.Cn_r = Cn_r
        self.Cm_q = Cm_q
        self.CL_elevator = CL_elevator
        self.Cm_elevator = Cm_elevator
        self.Cl_aileron = Cl_aileron
        self.Cn_rudder = Cn_rudder
        self.CY_rudder = CY_rudder


class PropConfig:
    def __init__(self, max_rpm: float = 8000.0, thrust_coef: float = 0.00001):
        self.max_rpm = max_rpm
        self.thrust_coef = thrust_coef


class ControlSigns:
    def __init__(self, aileron: float = 1.0, elevator: float = 1.0, rudder: float = -1.0, throttle: float = 1.0):
        self.aileron = aileron
        self.elevator = elevator
        self.rudder = rudder
        self.throttle = throttle


class DragConfig:
    def __init__(self, coeffs=(0.0, 0.0, 0.0)):
        self.coeffs = torch.tensor(coeffs)


def clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def calculate_propeller_thrust(throttle: torch.Tensor, max_thrust: float, prop: PropConfig) -> torch.Tensor:
    rpm = throttle * prop.max_rpm
    thrust = prop.thrust_coef * (rpm ** 2)
    thrust = torch.clamp(thrust, min=0.0, max=max_thrust)
    return torch.stack([thrust, torch.zeros_like(thrust), torch.zeros_like(thrust)], dim=-1)


def linear_drag(v_body: torch.Tensor, drag: DragConfig) -> torch.Tensor:
    coeffs = drag.coeffs.to(v_body.device)
    return -v_body * coeffs


def compute_aero_forces_and_moments(
    v_body: torch.Tensor,
    w_body: torch.Tensor,
    aileron: torch.Tensor,
    elevator: torch.Tensor,
    rudder: torch.Tensor,
    cfg: AeroConfig,
):
    v = v_body
    u = v[:, 0]
    vv = v[:, 1]
    w = v[:, 2]
    speed = torch.linalg.norm(v, dim=-1)
    safe_speed = torch.clamp(speed, min=0.1)

    alpha = torch.atan2(-w, u)
    beta = torch.asin(torch.clamp(-vv / safe_speed, min=-1.0, max=1.0))
    q_dyn = 0.5 * cfg.air_density * speed**2

    # Body rates are FLU; convert to FRD-style rates used by aero derivatives
    p = w_body[:, 0]
    q = -w_body[:, 1]
    r = -w_body[:, 2]
    p_hat = p * cfg.wing_span / (2.0 * safe_speed)
    q_hat = q * cfg.chord / (2.0 * safe_speed)
    r_hat = r * cfg.wing_span / (2.0 * safe_speed)

    CL_raw = cfg.CL_0 + cfg.CL_alpha * alpha + cfg.CL_elevator * elevator
    CL = torch.clamp(CL_raw, min=cfg.CL_min, max=cfg.CL_max)
    CD = cfg.CD_0 + cfg.CD_alpha * torch.abs(alpha) + cfg.CD_alpha2 * alpha**2
    CY = cfg.CY_beta * beta + cfg.CY_rudder * rudder

    L = CL * q_dyn * cfg.wing_area
    D = CD * q_dyn * cfg.wing_area
    Y = CY * q_dyn * cfg.wing_area

    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    fx_a = -D * cos_a + L * sin_a
    fy_a = Y
    fz_a = -D * sin_a - L * cos_a
    forces_flu = torch.stack([fx_a, -fy_a, -fz_a], dim=-1)

    Cl = cfg.Cl_beta * beta + cfg.Cl_aileron * aileron + cfg.Cl_p * p_hat + cfg.Cl_r * r_hat
    Cm = cfg.Cm_0 + cfg.Cm_alpha * alpha + cfg.Cm_elevator * elevator + cfg.Cm_q * q_hat
    Cn = cfg.Cn_beta * beta + cfg.Cn_rudder * rudder + cfg.Cn_p * p_hat + cfg.Cn_r * r_hat

    mx = Cl * q_dyn * cfg.wing_area * cfg.wing_span
    my = Cm * q_dyn * cfg.wing_area * cfg.chord
    mz = Cn * q_dyn * cfg.wing_area * cfg.wing_span
    moments_flu = torch.stack([mx, -my, -mz], dim=-1)

    mask = (speed >= 0.1).unsqueeze(-1)
    forces_flu = torch.where(mask, forces_flu, torch.zeros_like(forces_flu))
    moments_flu = torch.where(mask, moments_flu, torch.zeros_like(moments_flu))

    return forces_flu, moments_flu
