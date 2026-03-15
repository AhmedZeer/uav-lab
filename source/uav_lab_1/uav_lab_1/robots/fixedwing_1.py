import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uav_lab_1.robots import USD_PATHS

FIXEDWING_1_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS['fixedwing_1']),
    actuators={"propeller": ImplicitActuatorCfg(joint_names_expr=["propeller_joint"], damping=None, stiffness=None)},
)