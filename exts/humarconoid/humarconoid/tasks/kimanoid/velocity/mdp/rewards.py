from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    
from humarconoid.tasks.utils.tools import backlash


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


#########################################################################################
## CUSTOM REWARDS
#########################################################################################

def distance_btw_body(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), backlash_threshold: float = 3
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    distance = torch.linalg.norm(asset.data.body_state_w[:, 14] - asset.data.body_state_w[:, 15])

    if distance < backlash_threshold:
        distance = torch.tensor(0.0)

    return distance

def heel_toe_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    # compute the reward
    first_air_time_all = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg1.body_ids]
    first_air_time = torch.max(first_air_time_all, dim=1).values
    last_air_time_all = contact_sensor.data.last_air_time[:, sensor_cfg1.body_ids]
    last_air_time = torch.min(last_air_time_all, dim=1).values
    reward1 = last_air_time * first_air_time
    
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    # compute the reward
    first_air_time_all = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg2.body_ids]
    first_air_time = torch.max(first_air_time_all, dim=1).values
    last_air_time_all = contact_sensor.data.last_air_time[:, sensor_cfg2.body_ids]
    last_air_time = torch.min(last_air_time_all, dim=1).values
    reward2 = last_air_time * first_air_time
    
    reward = reward1 + reward2
    
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def heel_toe_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor1: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    contact_sensor2: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    # compute the reward    
    air_time1 = contact_sensor1.data.current_air_time[:, sensor_cfg1.body_ids]
    contact_time1 = contact_sensor1.data.current_contact_time[:, sensor_cfg1.body_ids]
    left_in_contact = torch.max(contact_time1, dim=1).values
    left_air_time = torch.min(air_time1, dim=1).values
    
    air_time2 = contact_sensor2.data.current_air_time[:, sensor_cfg2.body_ids]
    contact_time2 = contact_sensor2.data.current_contact_time[:, sensor_cfg2.body_ids]
    right_in_contact = torch.max(contact_time2, dim=1).values
    right_air_time = torch.min(air_time2, dim=1).values
    
    contact_time = torch.stack([left_in_contact, right_in_contact], dim=1)
    air_time = torch.stack([left_air_time, right_air_time], dim=1)

    in_contact = contact_time > 0.0
    
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def heel_heeltoe_toe_seq(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor1: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    contact_sensor2: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    # compute the reward    
    air_time1 = contact_sensor1.data.current_air_time[:, sensor_cfg1.body_ids]
    contact_time1 = contact_sensor1.data.current_contact_time[:, sensor_cfg1.body_ids]
    heel_in_contact = torch.all(contact_time1 > 0, dim=1)
    
    air_time2 = contact_sensor2.data.current_air_time[:, sensor_cfg2.body_ids]
    contact_time2 = contact_sensor2.data.current_contact_time[:, sensor_cfg2.body_ids]
    toe_in_contact = torch.all(contact_time2 > 0, dim=1)
    
    heel_contacted = heel_in_contact  # Heel is in contact
    toe_contacted = heel_contacted & toe_in_contact  # Toe contacts after heel
    
    # Reward is given only when the sequence heel -> toe is detected
    reward = torch.where(toe_contacted, torch.tensor(1.0, device=toe_contacted.device), torch.tensor(0.0, device=toe_contacted.device))
    
    # Clamp the reward to the threshold
    reward = torch.clamp(reward, max=1)
    
    # Apply command threshold (reward is zero if command is too small)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    return reward

def action_rate_l2_wo_leg(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions except knee & hip_pitch joints using L2 squared kernel."""
    exclude_indices = [6, 7, 9, 10] # 6, 7: knee / # 9, 10: hip_pitch
    
    mask = torch.ones(env.action_manager.action.shape[1], dtype=torch.bool, device=env.action_manager.action.device)
    mask[exclude_indices] = False
    
    action_diff = env.action_manager.action - env.action_manager.prev_action
    filtered_diff = action_diff[:, mask]
    
    return torch.sum(torch.square(filtered_diff), dim=1)

def heel_toe_motion_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float,
    heel_sensor_cfg1: SceneEntityCfg, toe_sensor_cfg1: SceneEntityCfg,
    heel_sensor_cfg2: SceneEntityCfg, toe_sensor_cfg2: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps with heel-toe motion for bipeds."""
    
    # 좌측 발의 힐과 토 센서 데이터
    heel_contact_sensor1: ContactSensor = env.scene.sensors[heel_sensor_cfg1.name]
    toe_contact_sensor1: ContactSensor = env.scene.sensors[toe_sensor_cfg1.name]
    heel_air_time1 = heel_contact_sensor1.data.current_air_time[:, heel_sensor_cfg1.body_ids]
    heel_contact_time1 = heel_contact_sensor1.data.current_contact_time[:, heel_sensor_cfg1.body_ids]
    toe_air_time1 = toe_contact_sensor1.data.current_air_time[:, toe_sensor_cfg1.body_ids]
    toe_contact_time1 = toe_contact_sensor1.data.current_contact_time[:, toe_sensor_cfg1.body_ids]
    
    # 우측 발의 힐과 토 센서 데이터
    heel_contact_sensor2: ContactSensor = env.scene.sensors[heel_sensor_cfg2.name]
    toe_contact_sensor2: ContactSensor = env.scene.sensors[toe_sensor_cfg2.name]
    heel_air_time2 = heel_contact_sensor2.data.current_air_time[:, heel_sensor_cfg2.body_ids]
    heel_contact_time2 = heel_contact_sensor2.data.current_contact_time[:, heel_sensor_cfg2.body_ids]
    toe_air_time2 = toe_contact_sensor2.data.current_air_time[:, toe_sensor_cfg2.body_ids]
    toe_contact_time2 = toe_contact_sensor2.data.current_contact_time[:, toe_sensor_cfg2.body_ids]

    # 힐-토 모션 체크
    left_heel_contacted = torch.all(heel_contact_time1 > 0, dim=1)
    left_toe_contacted = torch.all(toe_contact_time1 > 0, dim=1)
    right_heel_contacted = torch.all(heel_contact_time2 > 0, dim=1)
    right_toe_contacted = torch.all(toe_contact_time2 > 0, dim=1)
    
    # 힐-토 순서 확인 (힐이 먼저, 그 다음 토가 접촉)
    left_heel_toe_sequence = left_heel_contacted & left_toe_contacted
    right_heel_toe_sequence = right_heel_contacted & right_toe_contacted

    # 좌우 발의 힐-토 모션이 모두 성공한 경우
    heel_toe_motion = left_heel_toe_sequence & right_heel_toe_sequence

    # 기존 보상 계산
    contact_time = torch.stack([
        torch.max(heel_contact_time1, dim=1).values,
        torch.max(heel_contact_time2, dim=1).values
    ], dim=1)
    air_time = torch.stack([
        torch.min(heel_air_time1, dim=1).values,
        torch.min(heel_air_time2, dim=1).values
    ], dim=1)

    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    # 기존 보상과 힐-토 보상 결합
    base_reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    base_reward = torch.clamp(base_reward, max=threshold)

    # 힐-토 모션에 추가 보상 부여
    reward = base_reward + (heel_toe_motion.float() * 0.5)  # 힐-토 모션 시 추가 보상
    
    # 명령 크기가 충분하지 않으면 보상 없음
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    
    return reward

def action_rate_l2_leg(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the rate of change of the actions of knee & hip_pitch joints using L2 squared kernel."""
    include_indices = [6, 7, 9, 10] # 6, 7: knee / # 9, 10: hip_pitch
    
    mask = torch.ones(env.action_manager.action.shape[1], dtype=torch.bool, device=env.action_manager.action.device)
    mask[include_indices] = True
    
    action_diff = env.action_manager.action - env.action_manager.prev_action
    filtered_diff = action_diff[:, mask]
    
    return torch.sum(torch.square(filtered_diff), dim=1)


def leg_crossing_detection(
    env: ManagerBasedRLEnv, command_name: str, 
    left_sensor_cfg: SceneEntityCfg, right_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """Single support 상태에서 발의 속도 변화 패턴을 이용해 발 교차 여부를 판단."""
    
    left_contact_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right_contact_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    
    left_contact_time = left_contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids]
    right_contact_time = right_contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids]
    
    # 센서 데이터를 통해 왼발과 오른발의 접촉 여부 확인
    left_contact = torch.max(left_contact_time) > 0
    right_contact = torch.max(right_contact_time) > 0
    
    # Single Support 조건 확인 (한쪽 발만 접촉)
    single_support = (left_contact ^ right_contact)
    
    # asset = env.scene[asset_cfg.name]
    # body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    
    reward = 0
    
    # # 오른발의 x 방향 속도 가져오기 (왼발 기준)
    # right_foot_velocity = env.scene.sensors[right_sensor_cfg.name].data.velocity[:, 0]  # 오른발 x 속도
    
    # # 이전 프레임의 속도 저장 (이 부분은 상태 저장이 필요합니다)
    # if not hasattr(env, 'prev_right_foot_velocity'):
    #     env.prev_right_foot_velocity = right_foot_velocity.clone()

    # # 속도 변화 패턴 감지
    # velocity_change = (env.prev_right_foot_velocity * right_foot_velocity) < 0  # 부호 변화 감지

    # # Single support 상태에서 속도 변화가 있는 경우 발 교차로 판단
    # leg_crossed = single_support & velocity_change

    # # 현재 속도를 이전 속도로 업데이트
    # env.prev_right_foot_velocity = right_foot_velocity.clone()
    
    # reward = leg_crossed.float() * 0.5  # 교차 시 추가 보상
    # reward = torch.clamp(reward, max=1)
    
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    
    return reward