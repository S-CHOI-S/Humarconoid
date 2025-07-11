import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="G1-Wholebody",
    entry_point="humarconoid.envs:ARCManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1WholebodyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WholebodyFlatPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:G1WholebodyFlatPPORunnerCfg",
    },
)

gym.register(
    id="G1-Wholebody-Play",
    entry_point="humarconoid.envs:ARCManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1WholebodyFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WholebodyFlatPPORunnerCfg",
        "arc_rl_cfg_entry_point": f"{agents.__name__}.arc_rl_ppo_cfg:G1WholebodyFlatPPORunnerCfg",
    },
)
