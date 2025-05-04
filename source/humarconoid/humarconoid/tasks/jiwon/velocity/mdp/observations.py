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


def projected_gravity_body(
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


def body_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
        Get body heights related on the ground.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    body_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)

    return body_height


def gait_phase(env: ManagerBasedRLEnv, period: float = 2.5,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
        Get gait phase of the robot.
    """
    if hasattr(env, "episode_length_buf"):
        time = env.episode_length_buf * env.step_dt
    else:
        time = torch.zeros(env.num_envs, device=env.device)

    command_vel = env.command_manager.get_command("base_velocity")[:, :2]
    is_moving = torch.norm(command_vel, dim=1) > 0.2

    phase = time % period / period
    sin_phase = torch.sin(phase * 2 * math.pi)
    cos_phase = torch.cos(phase * 2 * math.pi)

    gait_phase = torch.stack([sin_phase, cos_phase], dim=-1)

    gait_phase[~is_moving] = 0.0

    return gait_phase
