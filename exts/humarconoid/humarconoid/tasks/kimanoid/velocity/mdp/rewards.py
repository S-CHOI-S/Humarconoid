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
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg
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
    reward = torch.clamp(reward, max=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.2 # 0.1
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
    exclude_indices = [6, 7, 9, 10] # 6, 7: hip_pitch / # 9, 10: knee
    
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
    left_heel_first = torch.all(heel_contact_time1 > toe_contact_time1, dim=1) & torch.all(toe_contact_time1 > 0, dim=1)
    right_heel_first = torch.all(heel_contact_time2 > toe_contact_time2, dim=1) & torch.all(toe_contact_time2 > 0, dim=1)
    
    left_heel_toe_sequence = left_heel_first
    right_heel_toe_sequence = right_heel_first
    
    # 좌우 발의 힐-토 모션이 올바른 순서로 성공한 경우
    heel_toe_motion = left_heel_toe_sequence & right_heel_toe_sequence
    
    # 기존 보상 계산
    contact_time = torch.stack([
        torch.maximum(heel_contact_time1, toe_contact_time1).squeeze(-1),
        torch.maximum(heel_contact_time2, toe_contact_time2).squeeze(-1)
    ], dim=1)
    air_time = torch.stack([
        torch.minimum(heel_air_time1, toe_air_time1).squeeze(-1),
        torch.minimum(heel_air_time2, toe_air_time2).squeeze(-1)
    ], dim=1)

    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time).squeeze(-1)
    
    # 양발이 번갈아 가며 접촉하도록 보상 추가
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # double_stance = torch.sum(in_contact.int(), dim=1) == 2
    fully_contact_reward = torch.zeros(single_stance.shape[0], device='cuda:0')
    
    if single_stance.any():  # single_stance가 하나라도 참인 환경이 있다면
        # print(f"single_stance: {single_stance}")
        # print(f"in_contact.int(): {in_contact.int()}")

        # 발뒤꿈치(heel)와 발끝(toe)의 접촉 시간 확인
        left_fully_contact = (in_contact[:, 0].unsqueeze(-1) == 1) & (heel_contact_time1 > 0) & (toe_contact_time1 > 0) # tensor([[False, False], [False, False]], device='cuda:0')
        right_fully_contact = (in_contact[:, 1].unsqueeze(-1) == 1) & (heel_contact_time2 > 0) & (toe_contact_time2 > 0) # tensor([[False, False], [False, False]], device='cuda:0')

        # print(f"in_contact[:, 0] == 1: {in_contact[:, 0].unsqueeze(-1) == 1}")
        # print(f"heel_contact_time1 > 0: {heel_contact_time1 > 0}")
        # print(f"toe_contact_time1 > 0: {toe_contact_time1 > 0}")
        # print(f"left_fully_contact or right_fully_contact: {torch.logical_or(left_fully_contact, right_fully_contact)}")
        
        # Combine left and right fully contact states
        fully_contact = torch.where(torch.logical_or(left_fully_contact, right_fully_contact), 0.2, 0)#torch.logical_or(left_fully_contact, right_fully_contact) # fully_contact: tensor([[False, False], [False, False]], device='cuda:0')


        # print(f"fully_contact: {fully_contact}")
        
        # print(f"[left_fully_contact, right_fully_contact]: {left_fully_contact}, {right_fully_contact}")
        

        # Reward tensor 초기화 (환경의 수만큼)
        # fully_contact_reward = torch.zeros(single_stance.shape[0], device='cuda:0')

        # fully_contact 조건이 True인 환경에 대해 reward 값을 1로 설정
        # print(fully_contact.squeeze(-1))
        fully_contact_reward = fully_contact.squeeze(-1)

        # print(f"reward: {reward}")
    
    # 한쪽 발만 계속 들고 있는 경우 페널티    
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    
    single_stance_reward = torch.where(single_stance, 1, 0.0) * (command_norm > 0.1) # tensor([0., 0., 0.], device='cuda:0')
    # threshold = torch.tensor(threshold, dtype=torch.float32, device=command_norm.device) # 1.0
    # print(f"single_stance_reward: {single_stance_reward}")
    # command_norm은 이미 텐서로 정의되어 있어야 합니다.
    threshold = torch.where(command_norm > 0.1, 1 / (2 * command_norm), torch.inf)

    # threshold가 텐서로 변환되도록 명확히 해줍니다.
    threshold = torch.tensor(threshold, device=command_norm.device)  # 텐서로 변환

    # unsqueeze(1)로 차원을 확장하여 air_time과 비교
    single_stance_penalty = torch.where(torch.any(air_time > threshold.unsqueeze(1), dim=1), -2, 0)
    
    # 보상 계산
    # base_reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # base_reward = torch.clamp(base_reward, max=threshold)
    
    # 힐-토 모션에 추가 보상 부여
    # heel_toe_motion == True: 1, False: 0
    reward = single_stance_reward + fully_contact_reward + single_stance_penalty + 0.25 * (left_heel_toe_sequence.float() + right_heel_toe_sequence.float()) # 힐-토 모션 시 추가 보상

    # 명령 크기가 충분하지 않으면 보상 없음
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    
    # print(f"single_stance_reward : {single_stance_reward}")
    # print(f"single_stance_penalty: {single_stance_penalty}")
    # print(f"heel_toe_motion      : {heel_toe_motion.float() * 0.5}")
    
    # print(f"single_stance_reward: {single_stance_reward[0]}")
    # print(f"single_stance_penalty: {single_stance_penalty[0]}")
    # print(f"air time: {air_time[0]}")
    # print(f"command norm: {command_norm[0]}")
    # print(f"threshold: {threshold[0]}")
    # print(f"left_heel_toe_sequence: {left_heel_toe_sequence.float()[0]}")
    # print(f"right_heel_toe_sequence: {right_heel_toe_sequence.float()[0]}")
    # print(f"reward: {reward[0]}")
    
    reward *= 0.15

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
    
    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    
    # 월드 좌표계 기준 발 위치 가져오기
    left_foot_pos_w = asset.data.body_pos_w[:, 13, :]
    right_foot_pos_w = asset.data.body_pos_w[:, 14, :]

    # 발 위치를 base frame 기준으로 변환
    left_foot_pos_b = quat_rotate_inverse(root_quat_w, left_foot_pos_w - root_pos_w)
    right_foot_pos_b = quat_rotate_inverse(root_quat_w, right_foot_pos_w - root_pos_w)
    
    relative_pos_x = left_foot_pos_b[:,0] - right_foot_pos_b[:,0]

    if not hasattr(env, 'prev_relative_pos_x'):
        env.prev_relative_pos_x = relative_pos_x.clone()

    # x 방향 상대 속도 부호 변화 감지
    position_sign_change = (env.prev_relative_pos_x * relative_pos_x) < 0  # 부호 변화 감지
    
    # Single support 상태에서 속도 부호 변화가 있을 경우 보상 부여
    reward = torch.where(single_support & position_sign_change, torch.tensor(1.0, device=relative_pos_x.device), torch.tensor(0.0, device=relative_pos_x.device))
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, 0], dim=1) > 0.1

    # 이전 속도를 현재 속도로 업데이트
    env.prev_relative_pos_x = relative_pos_x.clone()
    
    
    # left_foot_lin_vel_w = asset.data.body_lin_vel_w[:, 13, :]
    # right_foot_lin_vel_w = asset.data.body_lin_vel_w[:, 14, :]
    
    # # left_foot_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_names["Left_Leg6"], :]
    # # right_foot_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_names["Right_Leg6"], :]
    
    # left_foot_lin_vel_b = quat_rotate_inverse(root_quat_w, left_foot_lin_vel_w)
    # # left_foot_ang_vel_b = quat_rotate_inverse(root_quat_w, left_foot_ang_vel_w)
    # # left_foot_vel_b = torch.cat((left_foot_lin_vel_b, left_foot_ang_vel_b), dim=-1)
    
    # right_foot_lin_vel_b = quat_rotate_inverse(root_quat_w, right_foot_lin_vel_w)
    # # right_foot_ang_vel_b = quat_rotate_inverse(root_quat_w, right_foot_ang_vel_w)
    # # right_foot_vel_b = torch.cat((right_foot_lin_vel_b, right_foot_ang_vel_b), dim=-1)
    
    # relative_lin_vel_x = left_foot_lin_vel_b[:, 0] - right_foot_lin_vel_b[:, 0]
    
    # # 이전 x 방향 상대 속도 저장 변수 초기화
    # if not hasattr(env, 'prev_relative_lin_vel_x'):
    #     env.prev_relative_lin_vel_x = relative_lin_vel_x.clone()

    # # x 방향 상대 속도 부호 변화 감지
    # velocity_sign_change = (env.prev_relative_lin_vel_x * relative_lin_vel_x) < 0  # 부호 변화 감지
    
    # # Single support 상태에서 속도 부호 변화가 있을 경우 보상 부여
    # reward = torch.where(single_support & velocity_sign_change, torch.tensor(1.0, device=relative_lin_vel_x.device), torch.tensor(0.0, device=relative_lin_vel_x.device))
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    # # 이전 속도를 현재 속도로 업데이트
    # env.prev_relative_lin_vel_x = relative_lin_vel_x.clone()
    
    # print(f"left_foot_lin_vel_b:\n{left_foot_lin_vel_b[:5, 0]}")
    # print(f"right_foot_lin_vel_b:\n{right_foot_lin_vel_b[:5, 0]}")
    # print(f"relative_lin_vel_x:\n{relative_lin_vel_x[:5]}")
    # print(f"velocity_sign_change: {single_support & velocity_sign_change}")
    # print(f"leg_crossing_reward:\n{reward[:5]}")
    
    return reward

def get_phase(
    env: ManagerBasedRLEnv, command_name: str,
) -> torch.Tensor:
    
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    phase = env.episode_length_buf * (env.step_dt) * command_norm
    # print(env.episode_length_buf)
    # print(command_norm)
    # print(env.step_dt)
    
    return phase
    
def get_gait_phase(
    env: ManagerBasedRLEnv, command_name: str,
) -> torch.Tensor:
    
    phase = get_phase(env, command_name)
    sin_pos = torch.sin(2 * torch.pi * phase)
    
    # double support phase
    stance_mask = torch.zeros((env.num_envs, 2), device = sin_pos.device)
    
    # left foot stance
    stance_mask[:, 0] = sin_pos > 0
    
    # right foot stance
    stance_mask[:, 1] = sin_pos < 0
    
    return stance_mask

def ref_gait_phase(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    phase = get_phase(env, command_name)
    sin_pos = torch.sin(2 * torch.pi * phase)
    
    # compute reference state
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()
    
    left_joint_pos = asset.data.joint_pos[:, [6, 9, 11]]
    right_joint_pos = asset.data.joint_pos[:, [7, 10, 12]]
    ref_left_joint_pos = torch.zeros_like(left_joint_pos)
    ref_right_joint_pos = torch.zeros_like(right_joint_pos)
    left_pos_err = torch.zeros_like(left_joint_pos)
    right_pos_err = torch.zeros_like(right_joint_pos)
    
    scale = 0.5
    
    sin_pos_l[sin_pos_l > 0] = 0 # left swing (-)
    ref_left_joint_pos[:, 0] = -sin_pos_l * scale
    ref_left_joint_pos[:, 1] = -sin_pos_l * scale * 2
    ref_left_joint_pos[:, 2] = -sin_pos_l * scale
    
    sin_pos_r[sin_pos_l < 0] = 0 # right swing (+)
    ref_right_joint_pos[:, 0] = -sin_pos_r * scale
    ref_right_joint_pos[:, 1] = -sin_pos_r * scale * 2
    ref_right_joint_pos[:, 2] = -sin_pos_r * scale
    
    ref_left_joint_pos[torch.abs(sin_pos) < 0.05] = 0.
    ref_right_joint_pos[torch.abs(sin_pos) < 0.05] = 0.
    
    # print(f"\033[33mepisode_length_buf: \033[0m{env.episode_length_buf[20]}]")
    # print(f"\033[33mcommand_norm: \033[0m{command_norm[20]}]")
    # print(f"\033[33mphase: \033[0m{phase[20]}")
    # print(f"\033[33msin_pos: \033[0m{sin_pos[20]}")
    # if sin_pos[20] < 0:
    #     print(f"\033[35mleft feet: \033[0m{sin_pos[20]}")
    # elif sin_pos[20] > 0:
    #     print(f"\033[35mright feet: \033[0m{sin_pos[20]}")
    
    # print(f"\033[35mref_left_joint_pos: \033[0m{ref_left_joint_pos[20]}")
    # print(f"\033[34mleft_joint_pos: \033[0m{left_joint_pos[20]}")
    
    # print(f"\033[35mref_right_joint_pos: \033[0m{ref_right_joint_pos[20]}")
    # print(f"\033[34mright_joint_pos: \033[0m{right_joint_pos[20]}")

    valid_left_indices = ref_left_joint_pos != 0  # ref_left_joint_pos가 0이 아닌 위치
    valid_right_indices = ref_right_joint_pos != 0  # ref_right_joint_pos가 0이 아닌 위치

    left_pos_err[valid_left_indices] = ref_left_joint_pos[valid_left_indices] - left_joint_pos[valid_left_indices]
    right_pos_err[valid_right_indices] = ref_right_joint_pos[valid_right_indices] - right_joint_pos[valid_right_indices]

    # print(f"ref_gait = {-torch.sum(torch.abs(left_pos_err) + torch.abs(right_pos_err), dim=1)}")
    
    # Calculate the reward
    left_pos_norm = torch.norm(left_pos_err, dim=1)
    right_pos_norm = torch.norm(right_pos_err, dim=1)
    
    # left_reward = torch.exp(-2 * left_pos_norm) - left_pos_norm
    # right_reward = torch.exp(-2 * right_pos_norm) - right_pos_norm
    
    # Normalize to the range -0.5 to 0.5
    left_reward = torch.clamp(0.5 - left_pos_norm, min=-0.5, max=0.5)
    right_reward = torch.clamp(0.5 - right_pos_norm, min=-0.5, max=0.5)
    
    # Set reward to 0 where ref_joint_pos is 0
    left_reward = torch.where(torch.all(ref_left_joint_pos == 0, dim=1), torch.tensor(0.0, device=left_reward.device), left_reward)
    right_reward = torch.where(torch.all(ref_right_joint_pos == 0, dim=1), torch.tensor(0.0, device=right_reward.device), right_reward)
    
    total_reward = left_reward + right_reward
    
    # print(f"\033[34mleft:\033[0m {left_pos_err[20][1]} = {ref_left_joint_pos[20][1]} - {left_joint_pos[20][1]}")
    # print(f"\033[34mright:\033[0m {right_pos_err[20]}                      = {ref_right_joint_pos[20]} - {right_joint_pos[20]}")
    # print(f"\033[35mleft_reward:\033[0m {left_reward[20]}")
    # print(f"\033[35mright_reward:\033[0m {right_reward[20]}")
    # print(f"joint_limit[6],[9] : {asset.data.default_joint_limits[0][6]}, {asset.data.default_joint_limits[0][9]}")
    # print(f"joint_limit[7],[10]: {asset.data.default_joint_limits[0][7]}, {asset.data.default_joint_limits[0][10]}")
    
    return total_reward
    
    # return -torch.sum(torch.abs(left_pos_err) + torch.abs(right_pos_err), dim=1)
    
def feet_contact_number(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg,
) -> torch.Tensor:
    # sensor_cfg: 
    contact_sensor1: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    left_feet_contact = contact_sensor1.data.current_contact_time[:, sensor_cfg1.body_ids]
    contact_sensor2: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    right_feet_contact = contact_sensor2.data.current_contact_time[:, sensor_cfg2.body_ids]
    
    stance_mask = get_gait_phase(env, command_name).float()
    
    contact_status = torch.zeros(env.num_envs, dtype=torch.bool, device=stance_mask.device)
    penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=stance_mask.device)
    
    left_valid_indices = torch.where(stance_mask[:, 0] > 0)[0] # contacted left feet
    right_valid_indices = torch.where(stance_mask[:, 1] > 0)[0] # contacted right feet
    
    left_non_zero_indices = torch.where(
        (left_feet_contact[left_valid_indices, 0] != 0) | (left_feet_contact[left_valid_indices, 1] != 0)
    )[0]
    left_valid_indices = left_valid_indices[left_non_zero_indices]
    
    right_non_zero_indices = torch.where(
        (right_feet_contact[right_valid_indices, 0] != 0) | (right_feet_contact[right_valid_indices, 1] != 0)
    )[0]
    right_valid_indices = right_valid_indices[right_non_zero_indices]
    
    contact_status[left_valid_indices] = left_feet_contact[left_valid_indices, 0] >= left_feet_contact[left_valid_indices, 1]
    contact_status[right_valid_indices] = right_feet_contact[right_valid_indices, 0] >= right_feet_contact[right_valid_indices, 1]
    
    left_false_contact_indices = torch.where((stance_mask[:, 1] > 0) & 
                                         ((left_feet_contact[:, 0] != 0) | (left_feet_contact[:, 1] != 0)))[0]
    penalty[left_false_contact_indices] = -0.5
    
    right_false_contact_indices = torch.where((stance_mask[:, 0] > 0) & 
                                         ((right_feet_contact[:, 0] != 0) | (right_feet_contact[:, 1] != 0)))[0]
    penalty[right_false_contact_indices] = -0.5


    # print(contact_status)
    # print(contact_status.float())
    
    return contact_status.float() + penalty.float()