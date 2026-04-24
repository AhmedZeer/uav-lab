import gymnasium as gym

from . import agents


gym.register(
    id="Uav-Dogfight-2-Direct-v0",
    entry_point=f"{__name__}.uav_dogfight_env:UavDogfightEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.uav_dogfight_env_cfg:UavDogfightEnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
    },
)
