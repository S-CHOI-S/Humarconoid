from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import math
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
    # print(f"velocity command: {env.command_manager.get_command(command_name)[0, :2]}")
    # print(f"root lin vel: {asset.data.root_lin_vel_b[0, :2]}")
    # print(f"track_lin_vel: {torch.exp(-2 * lin_vel_error[0] / std**2)}")
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
#         is_stance = leg_phase[:, i] < 0.60

#         # contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
#         res += ~(contact ^ is_stance)
#     return res


def reward_feet_swing_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # contact_sensor1: ContactSensor = env.scene.sensors[sensor_cfg1.name]
    # left_feet_air = contact_sensor1.data.current_air_time[:, 25]
    # contact_sensor2: ContactSensor = env.scene.sensors[sensor_cfg2.name]
    # right_feet_air = contact_sensor2.data.current_air_time[:, 26]

    # left_air = left_feet_air > 0
    # right_air = right_feet_air > 0
    # only_one_air = left_air ^ right_air

    # left_air_indices = torch.nonzero(left_feet_air, as_tuple=True)[0]
    # right_air_indices = torch.nonzero(right_feet_air, as_tuple=True)[0]
    # only_one_air_indices = torch.nonzero(only_one_air, as_tuple=True)[0]

    left_foot_positions = asset.data.body_link_pos_w[:, 25, 2]
    right_foot_positions = asset.data.body_link_pos_w[:, 26, 2]

    # print(f"left_foot_positions rel: {left_foot_positions}")
    # print(f"right_foot_positions rel: {right_foot_positions}")

    gait_phase = gait_phase_from_obs(env)
    left_feet_swing = (gait_phase[:, 0] >= 0.60)
    right_feet_swing = (gait_phase[:, 1] >= 0.60)

    reward = torch.zeros(env.num_envs, device=env.device)

    # print("left_foot_positions shape:", left_foot_positions.shape)
    # print("left_foot_positions[left_mask] shape:", left_foot_positions[left_mask].shape)

    # left_error = torch.abs(left_foot_positions[left_feet_swing] - 0.25)
    # left_mask = left_error <= 0.06
    # reward[left_feet_swing] += (1.0 - left_error) * left_mask
    reward[left_feet_swing] -= torch.square(left_foot_positions[left_feet_swing] - 0.15)
    reward[right_feet_swing] -= torch.square(right_foot_positions[right_feet_swing] - 0.15)
    # print(f"left_foot_positions[left_feet_swing]: {left_foot_positions[left_feet_swing]}")
    # print(f"right_foot_positions[right_feet_swing]: {right_foot_positions[right_feet_swing]}")

    too_high_left = left_foot_positions[left_feet_swing] > 0.2
    reward[left_feet_swing][too_high_left] -= 1.0 * (left_foot_positions[left_feet_swing][too_high_left] + 1.2)

    too_high_right = right_foot_positions[right_feet_swing] > 0.2
    reward[right_feet_swing][too_high_right] -= 1.0 * (right_foot_positions[right_feet_swing][too_high_right] + 1.2)

    # right_error = torch.abs(right_foot_positions[right_feet_swing] - 0.25)
    # right_mask = right_error <= 0.06
    # reward[right_feet_swing] += (1.0 - right_error) * right_mask

    # contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    reward *= torch.where(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1, 1, 0)

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

    # # if step is long enough
    # exceed_air_time = air_time > 0.08
    # sufficient_air_time = torch.sum(exceed_air_time.int(), dim=1) == 0

    valid_air_time = (air_time >= 0.02) & (air_time <= 0.1)  # shape: [num_envs, 2]
    sufficient_air_time = torch.all(valid_air_time, dim=1)

    # print(f"single_stance\n: {single_stance[:5]}")
    # print(f"sufficient_air_time\n: {sufficient_air_time[:5]}")

    # get reward
    reward = torch.min(
        torch.where(
            single_stance.unsqueeze(-1),  # & sufficient_air_time.unsqueeze(-1) & contact_time_balance.unsqueeze(-1),
            in_mode_time,
            0.0,
        ),
        dim=1,
    )[0]
    reward = torch.clamp(reward, max=threshold)

    reward += torch.where(contact_time_balance, 0.1, 0)
    reward += torch.where(sufficient_air_time & single_stance, air_time.sum(dim=1) - 0.02, 0.0)

    # no reward for zero command
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.where(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1, 1, 0)

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


def gait_phase_from_obs(env: ManagerBasedRLEnv, stride_a: float = 8.0e-7, stride_b: float = 1.0,
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

    # 1. 속도 기반 stride length 계산
    stride_length = stride_a + stride_b * speed  # shape: [num_envs]
    stride_length *= 1.0  # scale stride length

    # 2. 주기 계산: T = L / v
    eps = 1e-7
    period = stride_length / (speed + eps)
    offset = 0.5

    phase = (time % period) / period
    left_phase = phase
    right_phase = (phase + offset) % 1
    gait_phase = torch.stack([left_phase, right_phase], dim=-1)

    gait_phase[~is_moving] = 0.0

    # print(f"{YELLOW}gait_phase: \n{RESET}{gait_phase}")

    return gait_phase


def symmetric_gait_phase(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Computes a gait symmetry reward.
    Includes:
    - phase-contact alignment per leg
    - left-right phase symmetry (π phase offset)
    - contact-stance synchronization between both legs
    - stability bonus when stationary

    Returns:
        torch.Tensor: shape [num_envs], per-environment reward.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # shape [num_envs, 2, 3]

    reward = torch.zeros(env.num_envs, device=env.device)

    # 1. Gait phase 가져오기
    gait_phase = gait_phase_from_obs(env)  # shape: [num_envs, 2]
    both_phase_stance = (gait_phase[:, 0] < 0.60) & (gait_phase[:, 1] < 0.60)

    # 2. 좌우 다리 각각의 phase-contact 일치 보상
    for i in range(2):
        is_stance = gait_phase[:, i] < 0.60
        force = net_forces_w[:, i, 2]
        contact = (force > 10) & (force < 200.0)

        mismatch = contact ^ is_stance
        reward += (~(mismatch)).float()

        # reward -= mismatch.float() * 0.75

    for i in range(2):
        is_swing = gait_phase[:, i] >= 0.60
        force = net_forces_w[:, i, 2]
        air = (force < 5)

        air_mismatch = air ^ is_swing
        reward += (~(air_mismatch)).float()

    mean_ep_length = env.episode_length_buf[0]
    # print(f"mean_ep_length: {mean_ep_length}")

    if mean_ep_length > 5000:
        for i in range(2):
            is_swing = gait_phase[:, i] >= 0.60
            force = net_forces_w[:, i, 2]
            contact = (force > 5)

            swing_contact_violation = is_swing & contact
            reward -= swing_contact_violation.float()

        for i in range(2):
            is_stance = gait_phase[:, i] < 0.60
            force = net_forces_w[:, i, 2]
            swing = (force <= 10)

            stance_swing_violation = is_stance & swing
            reward -= stance_swing_violation.float()

    # 3. 정지 상태 판단
    command_vel = env.command_manager.get_command(command_name)[:, :2]
    is_stationary = torch.norm(command_vel, dim=1) <= 0.1

    force_left = net_forces_w[:, 0, 2]
    force_right = net_forces_w[:, 1, 2]
    contact_left = (force_left > 10.0) & (force_left < 200.0)
    contact_right = (force_right > 10.0) & (force_right < 200.0)
    both_feet_contact = contact_left & contact_right

    # print(f"force_left: {force_left[0]}, force_right: {force_right[0]}")

    # 4. Stationary 보상: 정지 상태에서 두 발 모두 접촉 + stance 위상
    reward += (is_stationary & both_phase_stance & both_feet_contact).float() * 0.3

    # # === 이동 중일 때만 symmetry 보상 적용 ===
    # moving_mask = ~is_stationary
    # if moving_mask.any():
    #     sin_phase_left = gait_phase[:, 0]
    #     sin_phase_right = gait_phase[:, 1]

    #     # 좌우 π 위상 차이 유도
    #     symmetry_loss = (sin_phase_left + sin_phase_right)**2
    #     symmetry_reward = torch.exp(-symmetry_loss)  # [0,1] 범위
    #     reward[moving_mask] += symmetry_reward[moving_mask] * 0.35

    # 6. Contact-Stance 동기화 보상 (항상 적용)
    stance_left = (gait_phase[:, 0] < 0.60).float()
    stance_right = (gait_phase[:, 1] < 0.60).float()
    contact_left = contact_left.float()
    contact_right = contact_right.float()

    sync_error = torch.abs((contact_left - stance_left) - (contact_right - stance_right))
    sync_contact_reward = torch.exp(-10 * sync_error)  # 더 sharp하게 만듦
    reward += sync_contact_reward * 0.35

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


def symmetric_leg_phase(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for symmetric and wide leg motion using hip pitch joints.
    - Symmetric if: pos_L ≈ -pos_R and vel_L ≈ -vel_R
    - Wide if: |pos_L| large (max reward near 0.8 rad)

    Final reward = symmetry_reward * (1 + spread_weight * tanh(spread / scale))
    """
    # joint positions and velocities
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel

    # hip pitch joint indices
    lhpj_idx = 0
    rhpj_idx = 1

    pos_L = joint_pos[:, lhpj_idx]
    pos_R = joint_pos[:, rhpj_idx]
    vel_L = joint_vel[:, lhpj_idx]
    vel_R = joint_vel[:, rhpj_idx]

    gait_phase = gait_phase_from_obs(env)  # shape: [num_envs, 2]

    # 이동 중일 때만 적용
    command_vel = env.command_manager.get_command(command_name)[:, :2]
    cmd_speed = torch.norm(command_vel, dim=1)
    is_moving = cmd_speed > 0.1

    left_swing_phase = gait_phase[:, 0] >= 0.60
    right_swing_phase = gait_phase[:, 1] >= 0.60

    # --- Swing reward ---
    reward = torch.zeros_like(vel_L)

    left_mask = left_swing_phase & is_moving
    right_mask = right_swing_phase & is_moving

    reward[left_mask] += (-vel_L[left_mask]).clamp(min=0.0)
    reward[right_mask] += (-vel_R[right_mask]).clamp(min=0.0)

    # --- Double stance position reward ---
    double_stance = (gait_phase[:, 0] < 0.60) & (gait_phase[:, 1] < 0.60)

    phase_diff = gait_phase[:, 0] - gait_phase[:, 1]

    left_foot_pos_w = asset.data.body_link_state_w[:, 25, :3]     # link position in world frame
    right_foot_pos_w = asset.data.body_link_state_w[:, 26, :3]   # link position in world frame
    root_pos_w = asset.data.root_link_state_w[:, :3]         # root position in world frame
    root_quat_w = asset.data.root_link_state_w[:, 3:7]       # root orientation in world frame

    left_foot_pos_rel = root_pos_w[:, 0] - left_foot_pos_w[:, 0]
    right_foot_pos_rel = root_pos_w[:, 0] - right_foot_pos_w[:, 0]
    # left_foot_pos_b = quat_rotate_inverse(root_quat_w, left_foot_pos_rel)
    # right_foot_pos_b = quat_rotate_inverse(root_quat_w, right_foot_pos_rel)

    # print(f"left_foot_pos_rel: {left_foot_pos_rel}")
    # print(f"root_pos_w: {root_pos_w}")
    # print(f"right_foot_pos_rel: {right_foot_pos_rel}")

    # case1: right foot is in front of left foot
    # case1_mask = double_stance & (phase_diff > 0) & (left_foot_pos_rel > 0)
    # case1_reward = (left_foot_pos_rel - right_foot_pos_rel).clamp(min=-0.075) * torch.where(root_pos_w[:, 0] >= 0, 1, -1)
    # if case1_mask.float() > 0.0:
    #     print("\ncase1 ===================================")
    #     print("right_foot_pos_rel:", right_foot_pos_rel[case1_mask])
    #     print("left_foot_pos_rel:", left_foot_pos_rel[case1_mask])
    #     print("case1_reward:", (left_foot_pos_rel - right_foot_pos_rel).clamp(min=-0.075) * torch.where(root_pos_w[:, 0] >= 0, 1, -1))

    # case2: left foot is in front of right foot
    # case2_mask = double_stance & (phase_diff < 0) & (right_foot_pos_rel > 0)
    # case2_reward = (right_foot_pos_rel - left_foot_pos_rel).clamp(min=-0.075) * torch.where(root_pos_w[:, 0] >= 0, 1, -1)
    # if case2_mask.float() > 0.0:
    #     print("\ncase2 ===================================")
    #     print("right_foot_pos_rel:", right_foot_pos_rel[case2_mask])
    #     print("left_foot_pos_rel:", left_foot_pos_rel[case2_mask])
    #     print("case1_reward:", right_foot_pos_rel[case2_mask] - left_foot_pos_rel[case2_mask])

    # 총 리워드
    # position_reward = torch.zeros_like(left_foot_pos_rel)

    # case1_moving = case1_mask & is_moving
    # case2_moving = case2_mask & is_moving

    # position_reward[case1_moving] += case1_reward[case1_moving]  # * cmd_speed[case1_moving]
    # position_reward[case2_moving] += case2_reward[case2_moving]  # * cmd_speed[case2_moving]

    # symmetry error
    pos_error = torch.abs(pos_L + pos_R)
    vel_error = torch.abs(vel_L + vel_R)

    # symmetry reward
    alpha = 10.0
    beta = 0.25
    symmetry_error = pos_error + beta * vel_error
    symmetry_reward = torch.exp(-alpha * symmetry_error)

    # spread reward (soft saturate)
    spread = torch.abs((pos_L - pos_R) / 2)  # pos_R ≈ -pos_L
    spread_scale = 0.5 / 2  # tanh saturate 기준 (0.82 rad ≈ 0.96)
    spread_reward = torch.tanh(spread / spread_scale)

    # pos_error >= 0.2인 경우에는 reward를 0으로 만듦
    mask = pos_error >= 0.3
    spread_reward[mask] = 0.0

    # 최종 보상 조합
    spread_weight = 0.3
    final_reward = symmetry_reward * (1.0 + spread_weight * spread_reward)

    # 정지 시 0
    final_reward *= is_moving.float()

    return (reward + final_reward) * 10


def contact_velocity(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for contact velocity of feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Get the contact forces
    contact_forces = contact_sensor.data.net_forces_w[:, 25:27]  # shape: (N, B, 3)
    # Get the linear velocities of the feet
    feet_velocities = asset.data.body_lin_vel_w[:, 25:27, 2]  # shape: (N, B, 2)

    # Get the gait phase
    gait_phase = gait_phase_from_obs(env)  # shape: [num_envs, 2]

    penalty = torch.zeros(env.num_envs, device=env.device)

    left_foot = gait_phase[:, 0] >= 0.60  # left foot is almost finished swing phase
    right_foot = gait_phase[:, 1] >= 0.60  # right foot is almost finished swing phase

    penalty[left_foot] += feet_velocities[left_foot, 0].abs() ** 2  # penalize left foot velocity
    penalty[right_foot] += feet_velocities[right_foot, 1].abs() ** 2  # penalize right foot velocity

    for i in range(2):
        collide = contact_forces[:, i, 2] > 1200.0
        # Penalize the contact forces if they are above a threshold
        penalty[collide] += 5.0

    return penalty  # shape: (N,)


def base_height_l2_g1(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    penalty = torch.zeros(env.num_envs, device=env.device)

    # Compute the height penalty
    low_height = asset.data.root_pos_w[:, 2] < min_height
    penalty[low_height] += torch.abs(asset.data.root_pos_w[low_height, 2] - min_height)

    return penalty

def constant_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.episode_length_buf


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)