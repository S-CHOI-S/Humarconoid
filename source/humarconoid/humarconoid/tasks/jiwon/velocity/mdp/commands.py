from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from isaaclab.envs.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

class UniformVelocityCommandZ(CommandTerm):
    """Command generator that generates a velocity command in the z direction (vertical) only.

    The command comprises a linear velocity in the z direction. It is given in the robot's base frame.

    Mathematically, the velocity in the z direction is sampled as follows:

    .. math::

        v_z = \text{Uniform}(v_{\text{min}}, v_{\text{max}})
    """

    cfg: UniformVelocityCommandZCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandZCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # buffers to store the z-direction velocity command
        self.vel_command_b = torch.zeros(self.num_envs, 1, device=self.device)  # z-direction velocity only
        self.metrics["error_vel_z"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommandZ:\n"
        msg += f"\tCommand dimension: {self.command.shape[1:]}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the z direction. Shape is (num_envs, 1)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics to evaluate the performance of the z-direction command."""
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_z"] += (
            torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new velocity commands for the z direction."""
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_z)

    def _update_command(self):
        """Post-processes the z-direction velocity command."""
        pass  # No additional processing is required for z-direction only

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set visualization markers for the z-direction command."""
        if debug_vis:
            if not hasattr(self, "z_vel_visualizer"):
                self.z_vel_visualizer = VisualizationMarkers(self.cfg.z_vel_visualizer_cfg)
            self.z_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "z_vel_visualizer"):
                self.z_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback to visualize the z-direction velocity command."""
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_arrow_scale = torch.abs(self.command) * 3.0  # Scale visualization based on velocity
        self.z_vel_visualizer.visualize(base_pos_w, torch.tensor([0.0, 0.0, 0.0]), vel_arrow_scale)


@configclass
class UniformVelocityCommandZCfg(CommandTermCfg):
    """Configuration for the z-direction velocity command generator."""

    class_type: type = UniformVelocityCommandZ

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    resampling_time_range: tuple[float, float] = (0.5, 2.0)
    """Time range for resampling the z-direction velocity command."""

    @configclass
    class Ranges:
        """Uniform distribution range for the z-direction velocity command."""

        lin_vel_z: tuple[float, float] = (-1.0, 1.0)  # Min and max velocity in z direction

    ranges: Ranges = MISSING
    """Distribution ranges for the z-direction velocity commands."""

    z_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_z"
    )
    current_z_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_z_current"
    )
    """Configuration for the z-direction velocity visualization marker."""
    z_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_z_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
