# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import SceneEntityCfg

from humarconoid.envs import ARCManagerBasedEnv, ARCManagerBasedRLEnv
from torch import pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ARCManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ARCManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self.env = env

        self.amplitude = 0.5  # shoulder pitch amplitude

        self.l_shoulder_pitch_id = 11
        self.r_shoulder_pitch_id = 12

        self.l_shoulder_roll_id = 15
        self.r_shoulder_roll_id = 16

        self.l_elbow_id = 21
        self.r_elbow_id = 22

        self.l_wrist_roll_id = 23
        self.r_wrist_roll_id = 24

    # def apply_actions(self):
    #     # set position targets
    #     self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def apply_actions(self):
        leg_phase, speed = self.gait_phase(self.env)

        num_joints = self._asset.num_joints

        full_action = torch.zeros(self.num_envs, num_joints, device=self.device)

        if isinstance(self._joint_ids, slice):
            full_action[:, :] = self._processed_actions
        else:
            full_action[:, self._joint_ids] = self._processed_actions

        with torch.no_grad():
            pi_tensor = torch.pi
            arm_phase = (leg_phase + 0.5) % 1.0

            # damping 계산
            damping = torch.ones_like(leg_phase)
            in_ds1 = (leg_phase >= 0.0) & (leg_phase < 0.1)
            in_ds2 = (leg_phase >= 0.5) & (leg_phase < 0.6)

            damping[in_ds1] = 0.5 * (1 - torch.cos(pi_tensor * leg_phase[in_ds1] / 0.1))
            damping[in_ds2] = 0.5 * (1 + torch.cos(pi_tensor * (leg_phase[in_ds2] - 0.5) / 0.1))

            # speed-based modulation
            damping *= torch.clamp(speed / 0.2, 0.0, 1.0)

            # arm swing
            arm_angle = self.amplitude * damping * torch.sin(2 * pi_tensor * arm_phase)

            # joint target
            full_action[:, self.l_shoulder_pitch_id] = arm_angle + 0.3
            full_action[:, self.r_shoulder_pitch_id] = -arm_angle + 0.3
            full_action[:, self.l_elbow_id] = arm_angle + 0.97
            full_action[:, self.r_elbow_id] = -arm_angle + 0.97

            full_action[:, self.l_shoulder_roll_id] = 0.25
            full_action[:, self.r_shoulder_roll_id] = -0.25
            full_action[:, self.l_wrist_roll_id] = 0.15
            full_action[:, self.r_wrist_roll_id] = -0.15

        self._asset.set_joint_position_target(full_action)

    def gait_phase(self, env: ARCManagerBasedRLEnv, stride_a: float = 8.0e-7, stride_b: float = 1.0,
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

        stride_length = stride_a + stride_b * speed  # shape: [num_envs]
        stride_length *= 1.0  # scale stride length

        eps = 1e-7
        period = stride_length / (speed + eps)

        phase = (time % period) / period

        return phase, speed  # gait_phase
