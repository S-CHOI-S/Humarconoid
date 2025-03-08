from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def phase_information(env: ManagerBasedRLEnv) -> torch.Tensor:
    # phi = 
    # phase = 2 * torch.pi * phi
    # torch.sin()
    # env.step_dt
    return 1