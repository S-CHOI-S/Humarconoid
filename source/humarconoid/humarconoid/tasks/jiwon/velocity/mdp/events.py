# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_joint_position_noise(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    noise_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Applies random noise to the robot's joint positions.

    This function adds noise to the joint positions to simulate sensing errors or perturbations.
    It samples the noise for each joint from the given range dictionary and applies it to the simulation.

    The function takes a dictionary of noise ranges, where the keys are joint indices (as strings)
    and the values are tuples of the form ``(min, max)``.
    If a joint is not specified in the dictionary, no noise is added to that joint.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor): The environment IDs to apply the noise to.
        noise_range (dict[str, tuple[float, float]]): Noise range dictionary for each joint.
        asset_cfg (SceneEntityCfg): Configuration for the asset (robot).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[env_ids].clone()

    joint_pos += math_utils.sample_uniform(*noise_range, joint_pos.shape, joint_pos.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_joint_state_to_sim(joint_pos, asset.data.joint_vel[env_ids], env_ids=env_ids)


def randomize_scene_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: dict[str, tuple[float, float]],
):
    terrain: TerrainImporter = env.scene.terrain

    rand_friction = math_utils.sample_uniform(*friction_range, (2,), torch.device)

    terrain.cfg.physics_material.static_friction = rand_friction[0]
    terrain.cfg.physics_material.dynamic_friction = rand_friction[1]
