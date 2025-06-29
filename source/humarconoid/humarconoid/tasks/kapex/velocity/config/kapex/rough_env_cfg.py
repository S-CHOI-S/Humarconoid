from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import humarconoid.tasks.kapex.velocity.mdp as mdp

##
# Pre-defined configs
##
from humarconoid.robots import KAPEX_CFG
from humarconoid.tasks.kapex.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from humarconoid.tasks.utils import set_joint_mapping as KAPEX_JNT_CFG

joint_cfg = KAPEX_JNT_CFG.load_from_yaml("kapex")
# print("<><><><>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
# print("\tjoint_order:", joint_cfg["joint_order"])
# print("\tsorted_joint:", joint_cfg["sorted_joint"])
# print("<><><><>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
# joint_mapping = KAPEX_JNT_CFG.joint_mapping(joint_cfg["joint_order"], joint_cfg["sorted_joint"])
# # Output the mapping
# print("Joint Index Mapping:")
# for joint, indices in joint_mapping.items():
#     print(f"{joint}: joint_order_index={indices['original_index']}, sorted_joint_index={indices['sorted_index']}")


@configclass
class KapexRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # 1. [Penalty] Termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 2. [Reward] Tracking of Linear velocity Commands (xy axes)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 3. [Reward] Tracking of Angular Velocity Commands (yaw)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # 4. [Reward] Feet Air Time
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Leg[6-7]"),
    #         "threshold": 0.4,
    #     },
    # )

    # 5. [Penalty] Feet Slide
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Leg[6-7]"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Leg[6-7]"),
        },
    )

    # 6. [Penalty] Joint Limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[1-7]", "WJ[1-3]"])},
    )

    # # 7. [Penalty] Leg Joint Limits
    # dof_pos_limits_leg = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ4"])},
    # )

    # # 8. [Penalty] Torso Joint Limits
    # dof_pos_limits_torso = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["BWJ[1-3]"])},
    # )

    # 9. [Penalty] Deviation from default of the joints that are not essential for locomotion
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["WJ[1-3]"])},
    )

    # 10. [Penalty] Deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[1-2]"])},
    )

    # 11. [Penalty] Deviation from default of the joints that are not essential for locomotion
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.00,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[6-7]"])},
    )

    # # 11. [Reward] Heel-Toe Air Time
    # heel_toe_air_time = RewTerm(
    #     func=mdp.heel_toe_air_time_positive_biped,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg1": SceneEntityCfg("contact_forces", body_names="Left_Leg[6-7]"),
    #         "sensor_cfg2": SceneEntityCfg("contact_forces", body_names="Right_Leg[6-7]"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.4,
    #     },
    # )

    # 11. [Reward] Heel-Toe Motion Air Time
    # heel_toe_motion_air_time = RewTerm(
    #     func=mdp.heel_toe_motion_air_time_positive_biped,
    #     weight=0.125,
    #     params={
    #         "heel_sensor_cfg1": SceneEntityCfg("contact_forces", body_names="Left_Leg6"),
    #         "toe_sensor_cfg1": SceneEntityCfg("contact_forces", body_names="Left_Leg7"),
    #         "heel_sensor_cfg2": SceneEntityCfg("contact_forces", body_names="Right_Leg6"),
    #         "toe_sensor_cfg2": SceneEntityCfg("contact_forces", body_names="Right_Leg7"),
    #         "command_name": "base_velocity",
    #         "threshold": 1,
    #     }
    # )

    # 12. [Penalty] Reference Motion
    # reference_motion = RewTerm(
    #     func=mdp.ref_gait_phase,
    #     weight=0.15,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 1. Time Out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2. Base Link Contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Waist3"]), "threshold": 1.0},
    )


@configclass
class KapexRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KapexRewardsCfg = KapexRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = KAPEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
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

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*LJ[1-4]"])
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*LJ[1-7]"])

        ## Reshape Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -4e-05
        self.rewards.dof_acc_l2.weight = 0
        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_pos_limits.weight = -1
        # self.rewards.leg_crossing_detection.weight = 0
        self.rewards.termination_penalty.weight = -200.0
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_torso.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.1
        # self.rewards.heel_toe_air_time.weight = 1.5
        # self.rewards.reference_motion.weight = 0.25
        # self.rewards.contact_motion.weight = 1.25

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class KapexRoughEnvCfg_PLAY(KapexRoughEnvCfg):
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

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
