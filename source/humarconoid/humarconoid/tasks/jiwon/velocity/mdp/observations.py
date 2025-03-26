from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

from humarcscripts.color_code import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def flat_orientation_body(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
        Calculate non-flat body orientation using L2 squared kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)

    gravity = asset.data.GRAVITY_VEC_W

    # print(f"{RESET}quat: {quat.squeeze(1).size()}")
    # print(f"{RESET}quat: {quat.squeeze(1)}")
    # print(f"gravity: {gravity.size()}")
    # print(f"gravity: {gravity}")

    projected_grav = math_utils.quat_rotate_inverse(quat, gravity)

    return projected_grav


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
        Get joint torques applied on the articulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.applied_torque[:, asset_cfg.joint_ids]
