# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg

# if TYPE_CHECKING:
#     from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg

##
# Constraint manager
##


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    """Configuration for a constraint term."""

    func: Callable[..., None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input.
    """

    threshold: float | None = None         # 만족 여부 기준

    description: str = ""                  # (optional) 시각화 등에서 사용할 설명 텍스트

    modifiers: list[ModifierCfg] | None = None
    """The list of data modifiers to apply to the observation in order. Defaults to None,
    in which case no modifications will be applied.

    Modifiers are applied in the order they are specified in the list. They can be stateless
    or stateful, and can be used to apply transformations to the observation data. For example,
    a modifier can be used to normalize the observation data or to apply a rolling average.

    For more information on modifiers, see the :class:`~isaaclab.utils.modifiers.ModifierCfg` class.
    """

    noise: NoiseCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""

    clip: tuple[float, float] | None = None
    """The clipping range for the observation after adding noise. Defaults to None,
    in which case no clipping is applied."""

    scale: tuple[float, ...] | float | None = None
    """The scale to apply to the observation after clipping. Defaults to None,
    in which case no scaling is applied (same as setting scale to :obj:`1`).

    We leverage PyTorch broadcasting to scale the observation tensor with the provided value. If a tuple is provided,
    please make sure the length of the tuple matches the dimensions of the tensor outputted from the term.
    """

    history_length: int = 0
    """Number of past observations to store in the observation buffers. Defaults to 0, meaning no history.

    Observation history initializes to empty, but is filled with the first append after reset or initialization. Subsequent history
    only adds a single entry to the history buffer. If flatten_history_dim is set to True, the source data of shape
    (N, H, D, ...) where N is the batch dimension and H is the history length will be reshaped to a 2D tensor of shape
    (N, H*D*...). Otherwise, the data will be returned as is.
    """

    flatten_history_dim: bool = True
    """Whether or not the observation manager should flatten history-based observation terms to a 2D (N, D) tensor.
    Defaults to True."""


@configclass
class ConstraintGroupCfg:
    """Configuration for an constraint group."""

    concatenate_terms: bool = True
    """Whether to concatenate the constraint terms in the group. Defaults to True.

    If true, the constraint terms in the group are concatenated along the last dimension.
    Otherwise, they are kept separate and returned as a dictionary.

    If the constraint group contains terms of different dimensions, it must be set to False.
    """

    enable_corruption: bool = False
    """Whether to enable corruption for the constraint group. Defaults to False.

    If true, the constraint terms in the group are corrupted by adding noise (if specified).
    Otherwise, no corruption is applied.
    """

    history_length: int | None = None
    """Number of past constraint to store in the constraint buffers for all constraint terms in group.

    This parameter will override :attr:`ConstraintTermCfg.history_length` if set. Defaults to None. If None, each
    terms history will be controlled on a per term basis. See :class:`ConstraintTermCfg` for details on history_length
    implementation.
    """

    flatten_history_dim: bool = True
    """Flag to flatten history-based constraint terms to a 2D (num_env, D) tensor for all constraint terms in group.
    Defaults to True.

    This parameter will override all :attr:`ConstraintTermCfg.flatten_history_dim` in the group if
    constraintGroupCfg.history_length is set.
    """
