from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def phase_information(env: ManagerBasedRLEnv) -> torch.Tensor:
    # phi = 
    # phase = 2 * torch.pi * phi
    # torch.sin()
    # env.step_dt
    return 1