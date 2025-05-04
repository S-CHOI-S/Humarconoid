from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from humarcscripts.color_code import *


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    # print(f"velocity command: {env.command_manager.get_command(command_name)[:, :2]}")
    # print(f"root lin vel: {asset.data.root_lin_vel_b[:, :2]}")
    # print(f"track_lin_vel: {torch.exp(-lin_vel_error / std**2)}")
    return torch.exp(-2 * lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def flat_orientation_body(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    projected_gravity = quat_rotate_inverse(asset.data.body_link_quat_w[:, 11], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(projected_gravity), dim=1)


# def get_phase(
#     env: ManagerBasedRLEnv,
# ) -> torch.Tensor:
#     period = 0.8
#     offset = 0.5
#     dt = env.cfg.decimation * env.step_dt
#     phase = (env.episode_length_buf * dt) % period / period
#     phase_left = phase
#     phase_right = (phase + offset) % 1
#     leg_phase = torch.cat([phase_left.unsqueeze(1), phase_right.unsqueeze(1)], dim=-1)

#     return leg_phase


# def reward_contact(
#     env: ManagerBasedRLEnv,
# ) -> torch.Tensor:
#     res = torch.zeros(env.num_envs, dtype=torch.float32)
#     leg_phase = get_phase(env)
#     for i in range(self.feet_num):
#         is_stance = leg_phase[:, i] < 0.55

#         # contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
#         res += ~(contact ^ is_stance)
#     return res


def reward_feet_swing_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    contact_sensor1: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    left_feet_contact = contact_sensor1.data.current_air_time[:, sensor_cfg1.body_ids]
    contact_sensor2: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    right_feet_contact = contact_sensor2.data.current_air_time[:, sensor_cfg2.body_ids]

    left_air = left_feet_contact > 0
    right_air = right_feet_contact > 0
    only_one_air = left_air ^ right_air

    left_air_indices = torch.nonzero(left_air, as_tuple=True)[0]
    right_air_indices = torch.nonzero(right_air, as_tuple=True)[0]
    only_one_air_indices = torch.nonzero(only_one_air, as_tuple=True)[0]

    left_filtered_positions = asset.data.body_link_pos_w[left_air_indices, sensor_cfg1.body_ids, 2]
    right_filtered_positions = asset.data.body_link_pos_w[right_air_indices, sensor_cfg2.body_ids, 2]

    reward = torch.zeros(env.num_envs, device=left_feet_contact.device)

    left_mask = torch.isin(left_air_indices, only_one_air_indices)

    # print("left_filtered_positions shape:", left_filtered_positions.shape)
    # print("left_filtered_positions[left_mask] shape:", left_filtered_positions[left_mask].shape)

    reward[left_air_indices[left_mask]] += torch.norm(left_filtered_positions[left_mask] - 0.08)

    right_mask = torch.isin(right_air_indices, only_one_air_indices)
    reward[right_air_indices[right_mask]] += torch.norm(right_filtered_positions[right_mask] - 0.08)

    # contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    reward *= torch.where(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.2, 1, 0)

    return reward


def feet_safe_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,  # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # Contact forces for left foot
    contact_sensor1: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    left_feet_contact = contact_sensor1.data.current_contact_time[:, sensor_cfg1.body_ids]
    left_feet_contact_force = env.scene["contact_forces"].data.net_forces_w[:, sensor_cfg1.body_ids]  # Shape: (N, B, 3)
    # print(f"left_feet_contact_force: {left_feet_contact_force}")

    # Contact forces for right foot
    contact_sensor2: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    right_feet_contact = contact_sensor2.data.current_contact_time[:, sensor_cfg2.body_ids]
    right_feet_contact_force = env.scene["contact_forces"].data.net_forces_w[
        :, sensor_cfg2.body_ids
    ]  # Shape: (N, B, 3)
    # print(f"right_feet_contact_force: {right_feet_contact_force}")

    left_contact = left_feet_contact > 0
    right_contact = right_feet_contact > 0

    left_contact_indices = torch.nonzero(left_contact, as_tuple=True)[0]
    right_contact_indices = torch.nonzero(right_contact, as_tuple=True)[0]

    penalty = torch.zeros(env.num_envs, device=left_feet_contact.device)

    # Compute the norm of X, Y forces for each foot (Lateral forces)
    left_lateral_force = torch.norm(left_feet_contact_force[:, :, :1], dim=-1)  # sqrt(Fx^2 + Fy^2)
    right_lateral_force = torch.norm(right_feet_contact_force[:, :, :1], dim=-1)  # sqrt(Fx^2 + Fy^2)
    # print(f"{RESET}left_lateral_force: {left_feet_contact_force[0]}")
    # print(f"right_lateral_force: {right_feet_contact_force[0]}")

    # Compute penalty based on the lateral force magnitude > 0
    # Apply penalty only when there is contact
    if left_contact.any():
        penalty[left_contact.any(dim=1)] += left_lateral_force[left_contact].sum(dim=-1)

    if right_contact.any():
        penalty[right_contact.any(dim=1)] += right_lateral_force[right_contact].sum(dim=-1)
    # print(f"penalty: {penalty}")

    reward = 1.0 / (1.0 + penalty)

    return reward


def feet_air_time_balanced_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg  # balance_tolerance: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # current step
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # print(f"contact_time\n: {contact_time[:5]}")
    # print(f"air_time\n: {air_time[:5]}")

    # previous step
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # print(f"last_contact_time\n: {last_contact_time[:5]}")
    # print(f"last_air_time\n: {last_air_time[:5]}")

    # is current step contact
    in_contact = contact_time > 0.0
    # print(f"in_contact\n: {in_contact[:5]}")

    # which feet is contacted or in the air
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # print(f"in_mode_time\n: {in_mode_time[:5]}")

    # if single stance, get reward
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # print(f"single_stance\n: {single_stance[:5]}")

    # get reward if previous contact time & current contact time is similar
    time_diff = torch.zeros_like(contact_time[:, 0])

    time_diff = torch.where(  # left foot is in the air
        (single_stance.bool()) & (contact_time[:, 0] == 0),
        torch.abs(last_contact_time[:, 0] - contact_time[:, 1]),
        time_diff,
    )

    time_diff = torch.where(  # right foot is in the air
        (single_stance.bool()) & (contact_time[:, 1] == 0),
        torch.abs(contact_time[:, 0] - last_contact_time[:, 1]),
        time_diff,
    )

    # print(f"time_diff\n: {time_diff[:5]}")

    contact_time_balance = torch.where(time_diff < 0.1, 1, 0) == 1  # balance_tolerance
    # print(f"contact_time_balance\n: {contact_time_balance[:5]}")

    # if step is long enough
    exceed_air_time = air_time > 0.8
    sufficient_air_time = torch.sum(exceed_air_time.int(), dim=1) == 0
    # print(f"single_stance\n: {single_stance[:5]}")
    # print(f"sufficient_air_time\n: {sufficient_air_time[:5]}")

    # get reward
    reward = torch.min(
        torch.where(
            single_stance.unsqueeze(-1) & sufficient_air_time.unsqueeze(-1),  # & contact_time_balance.unsqueeze(-1),
            in_mode_time,
            0.0,
        ),
        dim=1,
    )[0]
    reward = torch.clamp(reward, max=threshold)

    # reward += torch.where(contact_time_balance, 0.1, 0)

    # no reward for zero command
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.where(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.2, 1, 0)

    return reward


def flat_orientation_feet(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    quat_11 = asset.data.body_link_quat_w[:, 11]
    quat_12 = asset.data.body_link_quat_w[:, 12]

    gravity = asset.data.GRAVITY_VEC_W

    projected_grav_11 = quat_rotate_inverse(quat_11, gravity)
    projected_grav_12 = quat_rotate_inverse(quat_12, gravity)
    # print(f"projected_grav_11: {projected_grav_11}")

    penalty_11 = torch.sum(torch.square(projected_grav_11), dim=1)
    penalty_12 = torch.sum(torch.square(projected_grav_12), dim=1)

    reward = 1.0 / (1.0 + penalty_11 + penalty_12)

    return reward


def symmetric_gait_phase(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:

    """Get gait phase of the robot.
    This function computes the gait phase of the robot based on the current episode length and the
    specified period and offset. The gait phase is represented as a tensor with two columns,
    representing the left and right leg phases, respectively.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # current step
    # air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    reward = torch.zeros(env.num_envs, device=env.device)

    gait_phase = env.observation_manager._obs_buffer["critic"][:, -2:]
    both_phase_stance = (gait_phase[:, 0] < 0.55) & (gait_phase[:, 1] < 0.55)

    for i in range(2):  # left and right leg
        is_stance = gait_phase[:, i] < 0.55
        force = net_forces_w[:, i, 2]
        contact = (force > 10) & (force < 250)
        reward += ~(contact ^ is_stance)
    # print(f"contact force: {net_forces_w[0, 0, 2]}, {net_forces_w[0, 1, 2]}")

    is_stationary = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) <= 0.2
    force_left = net_forces_w[:, 0, 2]
    force_right = net_forces_w[:, 1, 2]

    contact_left = (force_left > 10.0) & (force_left < 250.0)
    contact_right = (force_right > 10.0) & (force_right < 250.0)
    both_feet_contact = contact_left & contact_right

    reward *= torch.where(~is_stationary, 1, 0)

    reward += (is_stationary & both_phase_stance & both_feet_contact).float() * 0.12

    return reward


def undesired_pairwise_contact(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return 1 if bodyA and bodyB are in contact in each env, 0 otherwise."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    force_matrix = contact_sensor.data.force_matrix_w  # shape: (N, B, M, 3)

    # torch.set_printoptions(threshold=float('inf'))
    # # print("All sensor body IDs:", sensor_cfg.body_ids)
    # print("**********" * 5)
    # print("force matrix:\n", force_matrix[0].shape)  # pairwise_force_matrix[0]
    # print("force matrix:\n", force_matrix[0])
    # print(contact_sensor.cfg.__str__())
    # for name, sensor in env.scene.sensors.items():
    #     print(f"[Sensor] Name: {name}, Type: {type(sensor)}, Bodies: {sensor.body_names}")

    # print("Sensor bodies:", contact_sensor.data.entity_links_names)  # B
    # print("Contact targets:", contact_sensor.data.filter_prim_paths_expr_resolved)  # M

    # torch.set_printoptions(profile="default")

    # contact_force = torch.norm(force_matrix[:, sensor_cfg.body_ids[0], sensor_cfg.body_ids[1]], dim=-1)
    # is_contact = contact_force > threshold

    # 보통 reward에선 페널티로 -1 곱하거나, 단순 binary 반환
    return torch.zeros(env.num_envs, device=env.device)  # is_contact.float()  # 또는 -is_contact.float() for penalty
