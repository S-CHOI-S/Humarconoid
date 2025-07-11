from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def push_robot_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
):
    # obtain term settings
    term_cfg = env.event_manager.get_term_cfg("push_robot")

    cnt = 1
    mean_ep_length = torch.mean((env.episode_length_buf > (1 * env.max_episode_length / 2)).float())
    if mean_ep_length >= 0.5:
        cnt = min(2.5, 1 + 0.3 * ((mean_ep_length.item() - 0.5) // 0.1))
        term_cfg.params = {
            "velocity_range": {
                "x": (-0.2 * cnt, 0.2 * cnt),
                "y": (-0.2 * cnt, 0.2 * cnt),
                "z": (-0.2 * cnt, 0.2 * cnt),
                "roll": (-0.05 * cnt, 0.05 * cnt),
                "pitch": (-0.05 * cnt, 0.05 * cnt),
                "yaw": (-0.05 * cnt, 0.05 * cnt),
            }
        }
    else:
        term_cfg.params = {
            "velocity_range": {
                "x": (-0.2 * cnt, 0.2 * cnt),
                "y": (-0.2 * cnt, 0.2 * cnt),
                "z": (-0.2 * cnt, 0.2 * cnt),
                "roll": (-0.05 * cnt, 0.05 * cnt),
                "pitch": (-0.05 * cnt, 0.05 * cnt),
                "yaw": (-0.05 * cnt, 0.05 * cnt),
            }
        }
    env.event_manager.set_term_cfg("push_robot", term_cfg)
