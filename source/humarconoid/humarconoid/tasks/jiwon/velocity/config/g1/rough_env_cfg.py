from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import humarconoid.tasks.jiwon.velocity.mdp as mdp

##
# Pre-defined configs
##
from humarconoid.robots import G1_KIST_CFG
from humarconoid.tasks.jiwon.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg


@configclass
class JiwonRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_balanced_positive_biped,  ## ??
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_.*", ".*_ankle_.*"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_.*", ".*_hip_yaw_.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_.*", ".*_ankle_pitch_.*"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_.*"])},
    )
    flat_orientation_feet = RewTerm(
        func=mdp.flat_orientation_feet,
        weight=1.0,
    )

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #                 ".*_shoulder_roll_joint",
    #                 ".*_shoulder_yaw_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*",
    #             ],
    #         )
    #     },
    # )

    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_five_joint",
    #                 ".*_three_joint",
    #                 ".*_six_joint",
    #                 ".*_four_joint",
    #                 ".*_zero_joint",
    #                 ".*_one_joint",
    #                 ".*_two_joint",
    #             ],
    #         )
    #     },
    # )
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    # )

    # flat_orientation_body = RewTerm(func=mdp.flat_orientation_body, weight=0.0)

    feet_safe_contact = RewTerm(
        func=mdp.feet_safe_contact,
        weight=0.1,
        params={
            "sensor_cfg1": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "sensor_cfg2": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
        },
    )

    feet_swing_height = RewTerm(
        func=mdp.reward_feet_swing_height,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg1": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "sensor_cfg2": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*ankle_roll_link).*"]),
            "threshold": 1.0,
        },
    )


@configclass
class JiwonRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: JiwonRewards = JiwonRewards()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_KIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        # self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
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

        # self.actions.joint_pos.joint_names = {".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"}

        # Rewards
        self.rewards.undesired_contacts = None

        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.action_rate_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names={"^(?!.*_knee_).*"} # , ".*_knee_joint"]
        # )
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["^(?!.*_knee_).*"]  # , ".*_knee_joint"]
        )
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 1.0
        self.rewards.feet_slide.weight = -0.2
        self.rewards.dof_torques_l2.weight = -4.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_ankle_.*"]  # , ".*_knee_joint"]
        )
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names={"^(?!.*knee_joint).*"})
        # self.rewards.knee_action.weight = 0.0000000002
        # self.rewards.reward_feet_swing_height.weight = 0.003
        # self.rewards.undesired_contacts.weight =-1.0
        # self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        #     "robot", body_names=[".*_hip_.*", ".*_knee_joint",  ".*_ankle_.*"]
        # )
        # self.rewards.joint_deviation_torso.weight = -0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)


@configclass
class JiwonRoughEnvCfg_PLAY(JiwonRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
