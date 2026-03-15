import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg

from uav_lab_1.robots.fixedwing_1 import FIXEDWING_1_CONFIG

class BasicFixedWing1SceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(10000.0, 10000.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=5.0,
                dynamic_friction=10.0,
            ),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    uav = FIXEDWING_1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/fixedwing_1")
