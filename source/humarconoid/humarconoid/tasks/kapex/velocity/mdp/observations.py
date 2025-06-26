from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

import math
from humarcscripts.color_code import *


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
        Get joint torques applied on the articulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def gait_phase(env: ManagerBasedRLEnv, stride_a: float = 8.0e-7, stride_b: float = 1.0,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Computes the gait phase using speed-dependent stride length:
        L(v) = a + b * v
        period = L / v
    Returns sin/cos encoded phase.
    """
    if hasattr(env, "episode_length_buf"):
        time = env.episode_length_buf * env.step_dt
    else:
        time = torch.zeros(env.num_envs, device=env.device)

    command_vel = env.command_manager.get_command("base_velocity")[:, :2]
    speed = torch.norm(command_vel, dim=1)

    is_moving = speed > 0.1

    # 1. 속도 기반 stride length 계산
    stride_length = stride_a + stride_b * speed  # shape: [num_envs]
    stride_length *= 1.0  # scale stride length

    # 2. 주기 계산: T = L / v
    eps = 1e-7
    period = stride_length / (speed + eps)

    phase = (time % period) / period
    sin_phase = torch.sin(phase * 2 * math.pi)
    cos_phase = torch.cos(phase * 2 * math.pi)
    gait_phase = torch.stack([sin_phase, cos_phase], dim=-1)

    gait_phase[~is_moving] = 0.0

    # left_phase = phase
    # right_phase = (phase + 0.5) % 1

    # print(f"{RESET}\nspeed: {speed}")
    # if left_phase[0] >= 0.60 and right_phase[0] < 0.60:
    #     print(f"{RESET}gait_phase: \n{RESET} {RED}swing, {GREEN}stance")
    # if left_phase[0] >= 0.60 and right_phase[0] >= 0.60:
    #     print(f"{RESET}gait_phase: \n{RESET} {RED}swing, {RED}swing")
    # if left_phase[0] < 0.60 and right_phase[0] >= 0.60:
    #     print(f"{RESET}gait_phase: \n{RESET}{GREEN}stance, {RED}swing")
    # if left_phase[0] < 0.60 and right_phase[0] < 0.60:
    #     print(f"{RESET}gait_phase: \n{RESET}{GREEN}stance, {GREEN}stance")

    # print(f"{YELLOW}gait_phase: \n{RESET}{}")

    return gait_phase
