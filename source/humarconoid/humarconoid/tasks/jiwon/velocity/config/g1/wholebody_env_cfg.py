from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import RewardTermCfg as RewTerm
from humarconoid.tasks.jiwon.velocity.config.g1.rough_env_cfg import JiwonRewards

from isaaclab.managers import CurriculumTermCfg as CurrTerm

from .rough_env_cfg import JiwonRoughEnvCfg
import humarconoid.tasks.jiwon.velocity.mdp as mdp

from humarconoid.robots.g1_kist import G1_KIST_WHOLEBODY_CFG
from humarconoid.terrains.rough_slope import ROUGH_SLOPE_TERRAINS_WHOLEBODY_CFG

from humarconoid.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from humarconoid.tasks.jiwon.velocity.velocity_env_cfg import CurriculumCfg


@configclass
class JiwonWholebodyRewards(JiwonRewards):
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    flat_orientation_torso = RewTerm(func=mdp.flat_orientation_body, weight=-1.0)


@configclass
class JiwonWholebodyCurriculum(CurriculumCfg):
    push_robot_levels_ = CurrTerm(func=mdp.push_robot_levels)
    command_velocity_levels_ = CurrTerm(func=mdp.command_velocity_levels)


@configclass
class JiwonWholebodyEnvCfg(JiwonRoughEnvCfg):
    rewards: JiwonWholebodyRewards = JiwonWholebodyRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = G1_KIST_WHOLEBODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Observations
        # Policy
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            )
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            )
        }
        # Critic
        # self.observations.critic.base_lin_vel = None
        self.observations.critic.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            )
        }
        self.observations.critic.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            )
        }
        self.observations.critic.joint_torques.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            )
        }

        # Action
        self.actions.joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"], scale=1.0, use_default_offset=True)

        # Terrain
        self.scene.terrain.terrain_generator = ROUGH_SLOPE_TERRAINS_WHOLEBODY_CFG
        self.scene.terrain.max_init_terrain_level = 0

        # Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.75
        self.rewards.track_ang_vel_z_exp.weight = 3.25
        self.rewards.lin_vel_z_l2.weight = -0.0
        self.rewards.ang_vel_xy_l2.weight = -0.075

        self.rewards.dof_acc_l2.weight = -1.25e-06
        self.rewards.action_rate_l2.weight = -0.325

        self.rewards.feet_air_time.weight = 0.25

        self.rewards.flat_orientation_l2.weight = -4.0

        self.rewards.dof_pos_limits.weight = -1.0

        self.rewards.termination_penalty.weight = -300.0

        self.rewards.feet_slide.weight = -0.4

        self.rewards.joint_deviation_hip.weight = -1.0  # -0.5
        self.rewards.joint_deviation_ankle.weight = -1.0
        self.rewards.joint_deviation_knee.weight = -0.04

        self.rewards.symmetric_gait_phase.weight = 0.5
        self.rewards.symmetric_leg_phase.weight = 0.02

        # self.rewards.energy = None
        self.rewards.energy.weight = -1.0e-07
        self.rewards.flat_orientation_torso.weight = -0.5

        # Curriculum
        # self.curriculum.command_velocity_levels

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


class JiwonWholebodyEnvCfg_PLAY(JiwonWholebodyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = 0
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        # self.curriculum.push_robot_levels = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
