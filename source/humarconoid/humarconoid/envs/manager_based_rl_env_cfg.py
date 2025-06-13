# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


@configclass
class ARCManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a reinforcement learning environment with the ARC manager-based workflow."""

    # environment settings
    constraints: object | None = None
    """Constraint settings.

    Please refer to the :class:`humarconoid.managers.ConstraintManager` class for more details.
    """
