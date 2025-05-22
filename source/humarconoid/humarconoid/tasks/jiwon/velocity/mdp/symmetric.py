from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

import math
from humarcscripts.color_code import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def mirror_obs(obs: torch.Tensor, obs_type="policy") -> torch.Tensor:
    mirrored = obs.clone()

    joint_pos_start = 12
    joint_dof = 12

    left_idx = torch.tensor([0, 2, 4, 6, 8, 10], device=obs.device)
    right_idx = torch.tensor([1, 3, 5, 7, 9, 11], device=obs.device)

    # Joint pos
    pos_slice = mirrored[:, joint_pos_start : joint_pos_start + joint_dof]
    pos_slice[:, left_idx], pos_slice[:, right_idx] = \
        pos_slice[:, right_idx], pos_slice[:, left_idx]

    # # Joint vel
    # vel_start = joint_pos_start + joint_dof
    # vel_slice = mirrored[:, vel_start : vel_start + joint_dof]
    # vel_slice[:, left_idx], vel_slice[:, right_idx] = \
    #     vel_slice[:, right_idx], vel_slice[:, left_idx]

    return mirrored


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    mirrored = action.clone()
    left_idx = torch.tensor([0, 2, 4, 6, 8, 10], device=action.device)
    right_idx = torch.tensor([1, 3, 5, 7, 9, 11], device=action.device)

    mirrored[:, left_idx], mirrored[:, right_idx] = action[:, right_idx], action[:, left_idx]
    return mirrored
