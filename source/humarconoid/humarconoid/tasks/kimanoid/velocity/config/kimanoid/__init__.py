import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, stand_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Kimanoid-Stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.KimanoidStandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KimanoidStandPPORunnerCfg",
    },
)

gym.register(
    id="Kimanoid-Stand-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.KimanoidStandEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KimanoidStandPPORunnerCfg",
    },
)

gym.register(
    id="Kimanoid-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.KimanoidFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KimanoidFlatPPORunnerCfg",
        "sse_cfg_entry_point": f"{agents.__name__}.sse_cfg:KimanoidFlatSAEPPORunnerCfg",
    },
)

gym.register(
    id="Kimanoid-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.KimanoidFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KimanoidFlatPPORunnerCfg",
        "sse_cfg_entry_point": f"{agents.__name__}.sse_cfg:KimanoidFlatSAEPPORunnerCfg",
    },
)

gym.register(
    id="Kimanoid-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.KimanoidRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KimanoidRoughPPORunnerCfg",
        "sse_cfg_entry_point": f"{agents.__name__}.sse_cfg:KimanoidRoughSAEPPORunnerCfg",
    },
)

gym.register(
    id="Kimanoid-Rough-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.KimanoidRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KimanoidRoughPPORunnerCfg",
    },
)
