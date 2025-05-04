"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def modify_event_interval(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    interval_range: tuple[float, float],
    num_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Curriculum that modifies a event interval range a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        interval_range: The range of the event term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.interval_range_s = interval_range
        env.event_manager.set_term_cfg(term_name, term_cfg)


def push_robot_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # episode_length: int,
    # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # obtain term settings
    term_cfg = env.event_manager.get_term_cfg("push_robot")

    # if env.common_step_counter > 100000:
    #     # update term settings
    #     term_cfg.params = {
    #         "velocity_range": velocity_range
    #     }
    #     print("HERE: 1", env.episode_length_buf, env.max_episode_length / 3)
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
        # print("HERE: 3", env.common_step_counter, env.max_episode_length / 3)
    # print(f"cnt: {cnt}, {torch.mean((env.episode_length_buf > (env.max_episode_length / 4)).float())}")
    env.event_manager.set_term_cfg("push_robot", term_cfg)


def command_velocity_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # episode_length: int,
    # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # obtain term settings
    cmd_vel = env.command_manager._terms["base_velocity"]

    if env.common_step_counter < 3000:
        # print(f"HERE: {env.common_step_counter} < 10000")
        # update term settings
        for i in range(3):
            cmd_vel.vel_command_b[:, i] = 0
        # cmd_vel._update_command()

    # print(f"command_velocity: {cmd_vel.command[0]}")
