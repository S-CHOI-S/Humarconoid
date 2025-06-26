import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, stand_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Kapex-Stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.KapexStandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KapexStandPPORunnerCfg",
    },
)

gym.register(
    id="Kapex-Stand-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.KapexStandEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KapexStandPPORunnerCfg",
    },
)

gym.register(
    id="Kapex-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.KapexFlatEnvCfg,
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:KapexFlatPPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KapexFlatPPORunnerCfg",
        "sse_cfg_entry_point": f"{agents.__name__}.sse_cfg:KapexFlatSAEPPORunnerCfg",
    },
)

gym.register(
    id="Kapex-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.KapexFlatEnvCfg_PLAY,
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:KapexFlatPPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KapexFlatPPORunnerCfg",
        "sse_cfg_entry_point": f"{agents.__name__}.sse_cfg:KapexFlatSAEPPORunnerCfg",
    },
)

gym.register(
    id="Kapex-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.KapexRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KapexRoughPPORunnerCfg",
        "sse_cfg_entry_point": f"{agents.__name__}.sse_cfg:KapexRoughSAEPPORunnerCfg",
    },
)

gym.register(
    id="Kapex-Rough-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.KapexRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KapexRoughPPORunnerCfg",
    },
)
