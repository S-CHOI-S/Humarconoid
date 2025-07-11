from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import humarconoid.tasks.jiwon.velocity.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from humarconoid.robots import G1_KIST_CFG, G1_KIST_FLAT_FEET_CFG
from humarconoid.tasks.jiwon.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, MySceneCfg

from humarconoid.terrains.rough_slope import ROUGH_SLOPE_TERRAINS_CFG


@configclass
class JiwonScenes(MySceneCfg):
    contact_feet = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
                                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/right_ankle_roll_link"],
                                    history_length=3, track_air_time=True, force_threshold=0.0)
    # contact_knee = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_knee_link",
    #                                 filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/right_knee_link"],
    #                                 history_length=3, track_air_time=True, force_threshold=0.0)


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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_.*", ".*_hip_yaw_.*", ".*_hip_pitch_.*"])},
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
    # flat_orientation_feet = RewTerm(
    #     func=mdp.flat_orientation_feet,
    #     weight=1.0,
    # )

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

    # feet_safe_contact = RewTerm(
    #     func=mdp.feet_safe_contact,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg1": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
    #         "sensor_cfg2": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
    #     },
    # )

    feet_swing_height = RewTerm(
        func=mdp.reward_feet_swing_height,
        weight=0.1,
        params={
            "command_name": "base_velocity",
        },
    )

    symmetric_gait_phase = RewTerm(
        func=mdp.symmetric_gait_phase,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    symmetric_leg_phase = RewTerm(
        func=mdp.symmetric_leg_phase,
        weight=0.2,
        params={
            "command_name": "base_velocity",
        },
    )

    # undesired_pairwise_contact = RewTerm(
    #     func=mdp.undesired_pairwise_contact,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_feet"),
    #         "threshold": 0.5,
    #     },
    # )

    contact_velocity = RewTerm(
        func=mdp.contact_velocity,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
        },
    )

    base_height_l2_g1 = RewTerm(
        func=mdp.base_height_l2_g1,
        weight=-0.25,
        params={
            "min_height": 0.70,
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
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
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,
        },
    )


@configclass
class JiwonRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: JiwonScenes = JiwonScenes(num_envs=4096, env_spacing=2.5)
    rewards: JiwonRewards = JiwonRewards()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_KIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.scene.num_envs = 4096
        self.episode_length_s = 250.0  # max_episode_length = 15000

        # self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = ROUGH_SLOPE_TERRAINS_CFG
        # self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 2

        if self.scene.contact_feet is not None:
            self.scene.contact_feet.update_period = self.sim.dt

        # Events
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
        self.events.base_external_force_torque = None
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis"]
        self.events.reset_base.params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.07, 0.07),
                           "roll": (-0.25, 0.25), "pitch": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.25, 0.25),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        }
        # self.events.push_robot = None
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-2.5, 2.5),
                "y": (-2.5, 2.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            }
        }

        # Observations
        ## policy
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.02, n_max=0.02)
        # self.observations.policy.gait_phase = None
        ## critic
        self.observations.critic.height_scan = None
        self.observations.policy.base_lin_vel = None
        self.observations.critic.projected_gravity_feet1 = None
        self.observations.critic.projected_gravity_feet2 = None
        # self.observations.critic.gait_phase = None

        # self.actions.joint_pos.joint_names = ["^(?!.*_ankle_roll_).*"]

        # Rewards
        self.rewards.termination_penalty.weight = -300.0

        self.rewards.track_lin_vel_xy_exp.weight = 2.75
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 3.25
        self.rewards.lin_vel_z_l2.weight = -0.0
        self.rewards.ang_vel_xy_l2.weight = -0.075
        # self.rewards.dof_torques_l2 = None
        self.rewards.dof_torques_l2 = None  # -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        self.rewards.dof_acc_l2.weight = -1.25e-6
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*"]  # "^(?!.*_knee_).*"]
        )
        self.rewards.action_rate_l2.weight = -0.325
        # self.rewards.feet_air_time = None
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.feet_slide.weight = -0.4

        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_roll_.*", ".*_hip_yaw_.*"]
        )
        self.rewards.joint_deviation_ankle.weight = -1.0
        # self.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_ankle_roll_.*"]
        # )
        self.rewards.joint_deviation_knee.weight = -0.04

        # self.rewards.flat_orientation_feet = None
        # self.rewards.flat_orientation_feet.weight = 0.3

        # self.rewards.feet_safe_contact = None
        # self.rewards.feet_safe_contact.weight = 0.1
        # self.rewards.feet_swing_height = None
        self.rewards.feet_swing_height.weight = 0.0  # 0.15
        self.rewards.symmetric_gait_phase.weight = 0.5
        self.rewards.symmetric_leg_phase.weight = 0.02
        self.rewards.contact_velocity.weight = 0.0  # -0.75
        self.rewards.base_height_l2_g1.weight = -0.0
        self.rewards.base_height_l2_g1.params["min_height"] = 0.50

        self.rewards.undesired_contacts = None

        # Curriculums
        self.curriculum.push_robot_levels = None
        self.curriculum.command_velocity_levels = None
        # self.curriculum.command_velocity_levels.params

        # Terminations
        self.terminations.base_height = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


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

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.push_robot_levels = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
