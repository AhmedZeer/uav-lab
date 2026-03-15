# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Uav-Lab-1-Marl-Direct-v0",
    entry_point=f"{__name__}.uav_lab_1_marl_env:UavLab1MarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_lab_1_marl_env_cfg:UavLab1MarlEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)