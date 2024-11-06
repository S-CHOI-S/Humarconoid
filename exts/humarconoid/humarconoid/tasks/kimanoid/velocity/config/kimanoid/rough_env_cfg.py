from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import humarconoid.tasks.kimanoid.velocity.mdp as mdp
from humarconoid.tasks.kimanoid.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from humarconoid.robots import KIMANOID_CFG
from humarconoid.tasks.utils import set_joint_mapping as KIMANOID_JNT_CFG

joint_cfg = KIMANOID_JNT_CFG.load_from_yaml('kimanoid')
# print("<><><><>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
# print("\tjoint_order:", joint_cfg["joint_order"])
# print("\tsorted_joint:", joint_cfg["sorted_joint"])
# print("<><><><>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
# joint_mapping = KIMANOID_JNT_CFG.joint_mapping(joint_cfg["joint_order"], joint_cfg["sorted_joint"])
# # Output the mapping
# print("Joint Index Mapping:")
# for joint, indices in joint_mapping.items():
#     print(f"{joint}: joint_order_index={indices['original_index']}, sorted_joint_index={indices['sorted_index']}")

@configclass
class KimanoidRewardsCfg(RewardsCfg):
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
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Leg[6-7]"),
            "threshold": 0.4,
        },
    )
    
    # 5. [Penalty] Feet Slide
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Leg[6-7]"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Leg[6-7]"),
        },
    )

    # 6. [Penalty] Feet Joint Limits
    dof_pos_limits_feet = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[1-7]", "WJ[1-3]"])},
    )
    
    # 7. [Penalty] Leg Joint Limits
    # dof_pos_limits_leg = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[1-4]"])},
    # )
    
    # # 8. [Penalty] Torso Joint Limits
    # dof_pos_limits_torso = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["WJ[1-3]"])},
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


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 1. Time Out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 2. Base Link Contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Body_Waist3"]), "threshold": 1.0},
    )


@configclass
class KimanoidRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KimanoidRewardsCfg = KimanoidRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = KIMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/kimanoid"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["kimanoid"]
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
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*LJ[1-4]"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*LJ[1-7]"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class KimanoidRoughEnvCfg_PLAY(KimanoidRoughEnvCfg):
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
