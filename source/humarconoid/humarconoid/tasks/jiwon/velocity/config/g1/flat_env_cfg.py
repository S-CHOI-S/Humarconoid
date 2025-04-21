from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .rough_env_cfg import JiwonRoughEnvCfg


@configclass
class JiwonFlatEnvCfg(JiwonRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.num_envs = 8192  # 4096
        self.episode_length_s = 300.0  # max_episode_length = 15000

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Events
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
        self.events.base_external_force_torque = None
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # self.events.push_robot = None
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-5, 5),
                "y": (-3, 3),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        }

        # Observations
        ## policy
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.02, n_max=0.02)
        ## critic
        self.observations.critic.height_scan = None

        # self.actions.joint_pos.joint_names = ["^(?!.*_ankle_roll_).*"]

        # Rewards
        self.rewards.termination_penalty.weight = -1000.0

        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.dof_torques_l2 = None
        # self.rewards.dof_torques_l2.weight = -4.0e-6
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_ankle_.*", ".*_knee_.*"]  # , ".*_knee_joint"]
        # )
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["^(?!.*_knee_).*"]  # , ".*_knee_joint"]
        )
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.feet_air_time = None
        # self.rewards.feet_air_time.weight = 1.25
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.flat_orientation_l2.weight = -1.75
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.feet_slide.weight = -0.25

        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.joint_deviation_ankle.weight = -0.5
        self.rewards.joint_deviation_knee.weight = -0.05

        self.rewards.flat_orientation_feet = None
        # self.rewards.flat_orientation_feet.weight = 0.3

        self.rewards.feet_safe_contact = None
        # self.rewards.feet_safe_contact.weight = 0.1
        self.rewards.feet_swing_height = None
        # self.rewards.feet_swing_height.weight = 0.0075

        # Curriculums
        self.curriculum.push_robot_levels.params["velocity_range"] = {
            # "velocity_range": {"x": (-5.0, 5.0), "y": (-1.5, 1.5)}
            "x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (-1.0, 1.0),
            "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
        }

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)


class JiwonFlatEnvCfg_PLAY(JiwonFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.add_joint_noise = None
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }
        self.curriculum.push_robot_levels = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
