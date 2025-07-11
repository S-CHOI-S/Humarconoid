import gymnasium as gym

from . import agents, wholebody_env_cfg, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Jiwon-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.JiwonFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JiwonFlatPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:JiwonFlatPPORunnerCfg",
    },
)

gym.register(
    id="Jiwon-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.JiwonFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JiwonFlatPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:JiwonFlatPPORunnerCfg",
    },
)

gym.register(
    id="Jiwon-Rough",
    entry_point="humarconoid.envs:ARCManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.JiwonRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JiwonRoughPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:JiwonRoughPPORunnerCfg",
    },
)

gym.register(
    id="Jiwon-Rough-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.JiwonRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JiwonRoughPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:JiwonRoughPPORunnerCfg",
    },
)

gym.register(
    id="Jiwon-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wholebody_env_cfg.JiwonWholebodyEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JiwonWholebodyPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:JiwonWholebodyPPORunnerCfg",
    },
)

gym.register(
    id="Jiwon-Wholebody-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wholebody_env_cfg.JiwonWholebodyEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JiwonWholebodyPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:JiwonWholebodyPPORunnerCfg",
    },
)
