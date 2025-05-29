"""*******************************************************************************
* HumARConoid
*
* Advanced Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
*******************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from humarconoid.robots import HUMARCONOID_EXT_DIR

G1_KIST_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{HUMARCONOID_EXT_DIR}/G1/g1_29dof_rev_1_00/g1_29dof_rev_1_00.usd",
        usd_path=f"{HUMARCONOID_EXT_DIR}/G1/g1_kist.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.793),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,  # -0.28,
            ".*_knee_joint": 0.42,  # 0.6,
            ".*_ankle_pitch_joint": -0.23,  # -0.32,
            ".*_ankle_roll_joint": 0.0,
            # ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                # "waist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                # "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
                # "waist_.*_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                # "waist_.*_joint": 0.01,
            },
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "feet": DelayedPDActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_pitch_joint",
        #         ".*_shoulder_roll_joint",
        #         ".*_shoulder_yaw_joint",
        #         ".*_elbow_joint",
        #         ".*_wrist_.*",
        #     ],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness=40.0,
        #     damping=2.0,
        #     armature={
        #         ".*_shoulder_.*": 0.01,
        #         ".*_elbow_.*": 0.01,
        #         ".*_wrist_.*": 0.01,
        #     },
        # ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot (29dof)."""
"""
0  'pelvis', 'imu_in_pelvis', 'left_hip_pitch_link', 'pelvis_contour_link', 'right_hip_pitch_link',
5  'waist_yaw_link', 'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link', 'left_hip_yaw_link',
10 'right_hip_yaw_link', 'torso_link', 'left_knee_link', 'right_knee_link', 'd435_link', 'head_link',
15 'imu_in_torso', 'left_shoulder_pitch_link', 'logo_link', 'mid360_link', 'right_shoulder_pitch_link',
20 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link',
25 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link',
30 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link',
35 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link',
40 'left_rubber_hand', 'right_rubber_hand'
"""

"""
0  'left_hip_pitch_joint',
1  'right_hip_pitch_joint',
2  'left_hip_roll_joint',
3  'right_hip_roll_joint',
4  'left_hip_yaw_joint',
5  'right_hip_yaw_joint',
6  'left_knee_joint',
7  'right_knee_joint',
8  'left_ankle_pitch_joint',
9  'right_ankle_pitch_joint',
10 'left_ankle_roll_joint',
11 'right_ankle_roll_joint'
"""

# 0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 15, 16, 5, 11
# 0, 3, 6, 9, 11, 15, 1, 4, 7, 10, 12, 16, 2, 5, 8, 13, 14

# 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11
# 0, 2, 4, 6, 8, 10, 1, 3, 5, 9, 7, 11

G1_KIST_FLAT_FEET_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{HUMARCONOID_EXT_DIR}/G1/g1_29dof_rev_1_00/g1_29dof_rev_1_00.usd",
        usd_path=f"{HUMARCONOID_EXT_DIR}/G1/g1_kist.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.793),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,  # -0.28,
            ".*_knee_joint": 0.42,  # 0.6,
            ".*_ankle_pitch_joint": -0.23,  # -0.32,
            ".*_ankle_roll_joint": 0.0,
            # ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                # "waist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                # "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
                # "waist_.*_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                # "waist_.*_joint": 0.01,
            },
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "feet": DelayedPDActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_pitch_joint",
        #         ".*_shoulder_roll_joint",
        #         ".*_shoulder_yaw_joint",
        #         ".*_elbow_joint",
        #         ".*_wrist_.*",
        #     ],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness=40.0,
        #     damping=2.0,
        #     armature={
        #         ".*_shoulder_.*": 0.01,
        #         ".*_elbow_.*": 0.01,
        #         ".*_wrist_.*": 0.01,
        #     },
        # ),
    },
)