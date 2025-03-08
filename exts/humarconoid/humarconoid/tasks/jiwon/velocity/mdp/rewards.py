from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


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
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def flat_orientation_body(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    projected_gravity = quat_rotate_inverse(asset.data.body_link_quat_w[:, 11], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(projected_gravity), dim=1)

def get_phase(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    period = 0.8
    offset = 0.5
    dt = env.decimation * env.step_dt
    phase = (env.episode_length_buf * dt) % period / period
    phase_left = phase
    phase_right = (phase + offset) % 1
    leg_phase = torch.cat([phase_left.unsqueeze(1), phase_right.unsqueeze(1)], dim=-1)

    return leg_phase

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
    env: ManagerBasedRLEnv, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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

    left_mask = torch.isin(left_air_indices, only_one_air_indices)  # "하나만 공중"인 것만 선택
    
    # print("left_filtered_positions shape:", left_filtered_positions.shape)
    # print("left_filtered_positions[left_mask] shape:", left_filtered_positions[left_mask].shape)

    reward[left_air_indices[left_mask]] += torch.norm(left_filtered_positions[left_mask] - 0.08)

    right_mask = torch.isin(right_air_indices, only_one_air_indices)  # "하나만 공중"인 것만 선택
    reward[right_air_indices[right_mask]] += torch.norm(right_filtered_positions[right_mask] - 0.08)
    
    # contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    
    return reward

def feet_safe_contact(
    env: ManagerBasedRLEnv, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg, # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # Contact forces for left foot
    left_feet_contact_force = env.scene["contact_forces"].data.net_forces_w[:, sensor_cfg1.body_ids]  # Shape: (N, B, 3)

    # Contact forces for right foot
    right_feet_contact_force = env.scene["contact_forces"].data.net_forces_w[:, sensor_cfg2.body_ids]  # Shape: (N, B, 3)

    # Compute the norm of X, Y forces for each foot (Lateral forces)
    left_lateral_force = torch.norm(left_feet_contact_force[:, :, :2], dim=-1)  # sqrt(Fx^2 + Fy^2)
    right_lateral_force = torch.norm(right_feet_contact_force[:, :, :2], dim=-1)  # sqrt(Fx^2 + Fy^2)

    # Compute penalty based on the lateral force magnitude > 0
    penalty = (left_lateral_force + right_lateral_force).squeeze(-1)  # The larger the lateral force, the bigger the penalty
    
    return penalty