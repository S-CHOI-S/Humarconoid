# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint manager for computing constraint signals for a given world."""

from __future__ import annotations

import inspect
import numpy as np
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.utils import modifiers
from isaaclab.utils.buffers import CircularBuffer

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase

from humarconoid.managers.manager_term_cfg import ConstraintGroupCfg, ConstraintTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ConstraintManager(ManagerBase):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize constraint manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ConstraintGroupCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
            RuntimeError: If the shapes of the constraint terms in a group are not compatible for concatenation
                and the :attr:`~ConstraintGroupCfg.concatenate_terms` attribute is set to True.
        """
        # check that cfg is not None
        if cfg is None:
            raise ValueError("Constraint manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

        # compute combined vector for cstrnt group
        self._group_cstrnt_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        # self._cstrnt_term_threshold_values: dict[str, torch.Tensor] = {}
        for group_name, group_term_dims in self._group_cstrnt_term_dim.items():
            # if terms are concatenated, compute the combined shape into a single tuple
            # otherwise, keep the list of shapes as is
            if self._group_cstrnt_concatenate[group_name]:
                try:
                    term_dims = [torch.tensor(dims, device="cpu") for dims in group_term_dims]
                    self._group_cstrnt_dim[group_name] = tuple(torch.sum(torch.stack(term_dims, dim=0), dim=0).tolist())
                except RuntimeError:
                    raise RuntimeError(
                        f"Unable to concatenate constraint terms in group '{group_name}'."
                        f" The shapes of the terms are: {group_term_dims}."
                        " Please ensure that the shapes are compatible for concatenation."
                        " Otherwise, set 'concatenate_terms' to False in the group configuration."
                    )
            else:
                self._group_cstrnt_dim[group_name] = group_term_dims

        # Stores the latest constraints.
        self._cstrnt_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

    def __str__(self) -> str:
        """Returns: A string representation for the constraint manager."""
        msg = f"<ConstraintManager> contains {len(self._group_cstrnt_term_names)} groups.\n"

        # add info for each group
        for group_name, group_dim in self._group_cstrnt_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Constraint Terms in Group: '{group_name}'"
            if self._group_cstrnt_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"
            table.field_names = ["Index", "Name", "Shape"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            cstrnt_terms = zip(
                self._group_cstrnt_term_names[group_name],
                self._group_cstrnt_term_dim[group_name],
            )
            for index, (name, dims) in enumerate(cstrnt_terms):
                # resolve inputs to simplify prints
                tab_dims = tuple(dims)
                # add row
                table.add_row([index, name, tab_dims])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        terms = []

        if self._cstrnt_buffer is None:
            self.compute()
        cstrnt_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._cstrnt_buffer

        for group_name, _ in self._group_cstrnt_dim.items():
            if not self.group_cstrnt_concatenate[group_name]:
                for name, term in cstrnt_buffer[group_name].items():
                    terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
                continue

            idx = 0
            # add info for each term
            data = cstrnt_buffer[group_name]
            for name, shape in zip(
                self._group_cstrnt_term_names[group_name],
                self._group_cstrnt_term_dim[group_name],
            ):
                data_length = np.prod(shape)
                term = data[env_idx, idx : idx + data_length]
                terms.append((group_name + "-" + name, term.cpu().tolist()))
                idx += data_length

        return terms

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active constraint terms in each group.

        The keys are the group names and the values are the list of constraint term names in the group.
        """
        return self._group_cstrnt_term_names

    @property
    def group_cstrnt_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        """Shape of computed constraints in each group.

        The key is the group name and the value is the shape of the constraint tensor.
        If the terms in the group are concatenated, the value is a single tuple representing the
        shape of the concatenated constraint tensor. Otherwise, the value is a list of tuples,
        where each tuple represents the shape of the constraint tensor for a term in the group.
        """
        return self._group_cstrnt_dim

    @property
    def group_cstrnt_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        """Shape of individual constraint terms in each group.

        The key is the group name and the value is a list of tuples representing the shape of the constraint terms
        in the group. The order of the tuples corresponds to the order of the terms in the group.
        This matches the order of the terms in the :attr:`active_terms`.
        """
        return self._group_cstrnt_term_dim

    @property
    def group_cstrnt_concatenate(self) -> dict[str, bool]:
        """Whether the constraint terms are concatenated in each group or not.

        The key is the group name and the value is a boolean specifying whether the constraint terms in the group
        are concatenated into a single tensor. If True, the constraints are concatenated along the last dimension.

        The values are set based on the :attr:`~ConstraintGroupCfg.concatenate_terms` attribute in the group
        configuration.
        """
        return self._group_cstrnt_concatenate

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for group_name, group_cfg in self._group_cstrnt_class_term_cfgs.items():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
            # reset terms with history
            for term_name in self._group_cstrnt_term_names[group_name]:
                if term_name in self._group_cstrnt_term_history_buffer[group_name]:
                    self._group_cstrnt_term_history_buffer[group_name][term_name].reset(batch_ids=env_ids)
        # call all modifiers that are classes
        for mod in self._group_cstrnt_class_modifiers:
            mod.reset(env_ids=env_ids)

        # nothing to log here
        return {}

    def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the constraints per group for all groups.

        The method computes the constraints for all the groups handled by the constraint manager.
        Please check the :meth:`compute_group` on the processing of constraints per group.

        Returns:
            A dictionary with keys as the group names and values as the computed constraints.
            The constraints are either concatenated into a single tensor or returned as a dictionary
            with keys corresponding to the term's name.
        """
        # create a buffer for storing cstrnt from all the groups
        cstrnt_buffer = dict()
        # iterate over all the terms in each group
        for group_name in self._group_cstrnt_term_names:
            cstrnt_buffer[group_name] = self.compute_group(group_name)
        # otherwise return a dict with constraints of all groups

        # Cache the constraints.
        self._cstrnt_buffer = cstrnt_buffer
        return cstrnt_buffer

    def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the constraints for a given group.

        The constraints for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each constraint term:

        1. Compute constraint term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ConstraintTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ConstraintTermCfg.noise`
        4. Apply clipping based on :attr:`ConstraintTermCfg.clip`
        5. Apply scaling based on :attr:`ConstraintTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the constraints. Defaults to None,
                in which case constraints for all the groups are computed and returned.

        Returns:
            Depending on the group's configuration, the tensors for individual constraint terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check ig group name is valid
        if group_name not in self._group_cstrnt_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the constraint manager."
                f" Available groups are: {list(self._group_cstrnt_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_cstrnt_term_names[group_name]
        # buffer to store cstrnt per group
        group_cstrnt = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        cstrnt_terms = zip(group_term_names, self._group_cstrnt_term_cfgs[group_name])

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in cstrnt_terms:
            # compute term's value
            cstrnt: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    cstrnt = modifier.func(cstrnt, **modifier.params)
            if term_cfg.noise:
                cstrnt = term_cfg.noise.func(cstrnt, term_cfg.noise)
            if term_cfg.clip:
                cstrnt = cstrnt.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                cstrnt = cstrnt.mul_(term_cfg.scale)
            # Update the history buffer if constraint term has history enabled
            if term_cfg.history_length > 0:
                self._group_cstrnt_term_history_buffer[group_name][term_name].append(cstrnt)
                if term_cfg.flatten_history_dim:
                    group_cstrnt[term_name] = self._group_cstrnt_term_history_buffer[group_name][term_name].buffer.reshape(
                        self._env.num_envs, -1
                    )
                else:
                    group_cstrnt[term_name] = self._group_cstrnt_term_history_buffer[group_name][term_name].buffer
            else:
                group_cstrnt[term_name] = cstrnt

        # concatenate all constraints in the group together
        if self._group_cstrnt_concatenate[group_name]:
            return torch.cat(list(group_cstrnt.values()), dim=-1)
        else:
            return group_cstrnt

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: TerminationTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the termination term.
            cfg: The configuration for the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Termination term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the termination term.

        Returns:
            The configuration of the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Termination term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of constraint terms functions."""
        # create buffers to store information for each constraint group
        # TODO: Make this more convenient by using data structures.
        self._group_cstrnt_term_names: dict[str, list[str]] = dict()
        self._group_cstrnt_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_cstrnt_term_cfgs: dict[str, list[ConstraintTermCfg]] = dict()
        self._group_cstrnt_class_term_cfgs: dict[str, list[ConstraintTermCfg]] = dict()
        self._group_cstrnt_concatenate: dict[str, bool] = dict()
        self._group_cstrnt_term_history_buffer: dict[str, dict] = dict()
        # create a list to store modifiers that are classes
        # we store it as a separate list to only call reset on them and prevent unnecessary calls
        self._group_cstrnt_class_modifiers: list[modifiers.ModifierBase] = list()

        # make sure the simulation is playing since we compute cstrnt dims which needs asset quantities
        if not self._env.sim.is_playing():
            raise RuntimeError(
                "Simulation is not playing. Constraint manager requires the simulation to be playing"
                " to compute constraint dimensions. Please start the simulation before using the"
                " constraint manager."
            )

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, ConstraintGroupCfg):
                raise TypeError(
                    f"Constraint group '{group_name}' is not of type 'ConstraintGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_cstrnt_term_names[group_name] = list()
            self._group_cstrnt_term_dim[group_name] = list()
            self._group_cstrnt_term_cfgs[group_name] = list()
            self._group_cstrnt_class_term_cfgs[group_name] = list()
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()
            # read common config for the group
            self._group_cstrnt_concatenate[group_name] = group_cfg.concatenate_terms
            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            for term_name, term_cfg in group_cfg_items:
                # skip non-cstrnt settings
                if term_name in ["enable_corruption", "concatenate_terms", "history_length", "flatten_history_dim"]:
                    continue
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ConstraintTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)

                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # check group history params and override terms
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                # add term config to list to list
                self._group_cstrnt_term_names[group_name].append(term_name)
                self._group_cstrnt_term_cfgs[group_name].append(term_cfg)

                # register externally-set threshold
                # if term_cfg.threshold is not None:
                #     self._cstrnt_term_threshold_values[f"{group_name}/{term_name}"] = torch.tensor(
                #         term_cfg.threshold, dtype=torch.float, device=self._env.device
                #     )

                # call function the first time to fill up dimensions
                cstrnt_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)

                # create history buffers and calculate history term dimensions
                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                    )
                    old_dims = list(cstrnt_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    cstrnt_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        cstrnt_dims = (cstrnt_dims[0], np.prod(cstrnt_dims[1:]))

                self._group_cstrnt_term_dim[group_name].append(cstrnt_dims[1:])

                # if scale is set, check if single float or tuple
                if term_cfg.scale is not None:
                    if not isinstance(term_cfg.scale, (float, int, tuple)):
                        raise TypeError(
                            f"Scale for constraint term '{term_name}' in group '{group_name}'"
                            f" is not of type float, int or tuple. Received: '{type(term_cfg.scale)}'."
                        )
                    if isinstance(term_cfg.scale, tuple) and len(term_cfg.scale) != cstrnt_dims[1]:
                        raise ValueError(
                            f"Scale for constraint term '{term_name}' in group '{group_name}'"
                            f" does not match the dimensions of the constraint. Expected: {cstrnt_dims[1]}"
                            f" but received: {len(term_cfg.scale)}."
                        )

                    # cast the scale into torch tensor
                    term_cfg.scale = torch.tensor(term_cfg.scale, dtype=torch.float, device=self._env.device)

                # prepare modifiers for each constraint
                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with constraint size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for constraint term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=cstrnt_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_cstrnt_class_modifiers.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of constraint term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of constraint term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first two arguments for env and env_ids
                        # Think: Check for cases when kwargs are set inside the function?
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of constraint term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )

                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_cstrnt_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get cstrnt dims changed the state)
                    term_cfg.func.reset()
            # add history buffers for each group
            self._group_cstrnt_term_history_buffer[group_name] = group_entry_history_buffer

    # def set_threshold(self, group_name: str, term_name: str, threshold: float | torch.Tensor):
    #     key = f"{group_name}/{term_name}"
    #     if isinstance(threshold, (float, int)):
    #         threshold = torch.tensor(threshold, dtype=torch.float, device=self._env.device)
    #     elif isinstance(threshold, torch.Tensor):
    #         threshold = threshold.to(self._env.device).float()
    #     else:
    #         raise TypeError(f"Unsupported type for threshold: {type(threshold)}")

    #     self._cstrnt_term_threshold_values[key] = threshold

    # def get_threshold(self, group_name: str, term_name: str) -> torch.Tensor | None:
    #     return self._cstrnt_term_threshold_values.get(f"{group_name}/{term_name}", None)

    def get_group_threshold_tensor(self, group_name: str) -> torch.Tensor:
        """
        Returns the concatenated threshold tensor for a constraint group.

        This returns a 1D tensor of size (total_dims,) representing the threshold per term dimension.

        Args:
            group_name: Name of the group.

        Returns:
            Concatenated threshold tensor (1D).
        """
        print("///////////////////////////////////////////////////////////////////////")
        if group_name not in self._group_cstrnt_term_names:
            raise ValueError(f"Group '{group_name}' not found in constraint manager.")

        threshold_list = []

        for term_name, dims, cfg in zip(
            self._group_cstrnt_term_names[group_name],
            self._group_cstrnt_term_dim[group_name],
            self._group_cstrnt_term_cfgs[group_name],
        ):
            term_dim = int(np.prod(dims))
            threshold = cfg.threshold

            if threshold is None:
                # default to +inf to avoid violating
                t = torch.full((term_dim,), float("inf"), device=self._env.device)
            elif isinstance(threshold, (float, int)):
                t = torch.full((term_dim,), threshold, device=self._env.device)
            elif isinstance(threshold, torch.Tensor):
                if threshold.numel() != term_dim:
                    raise ValueError(
                        f"Threshold for term '{term_name}' in group '{group_name}' does not match term dimension. "
                        f"Expected {term_dim}, got {threshold.numel()}."
                    )
                t = threshold.to(device=self._env.device)
            else:
                raise TypeError(f"Unsupported threshold type for term '{term_name}': {type(threshold)}")

            threshold_list.append(t)

        return torch.cat(threshold_list, dim=0)
