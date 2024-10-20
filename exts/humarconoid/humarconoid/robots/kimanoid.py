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

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from humarconoid.robots import HUMARCONOID_EXT_DIR

KIMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{HUMARCONOID_EXT_DIR}/KIST_HUMANOID_TORSO/kist_humanoid3.usd",
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
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    
    soft_joint_pos_limit_factor=0.9,
    
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "LLJ[1-4]",
                "RLJ[1-4]",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "LLJ[1-4]": 150.0,
                "RLJ[1-4]": 150.0,
            },
            damping={
                "LLJ[1-4]": 5.0,
                "RLJ[1-4]": 5.0,
            },
            armature={
                "LLJ[1-4]": 0.01,
                "RLJ[1-4]": 0.01,
            },
        ),
        
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=["LLJ[5-7]", "RLJ[5-7]"],
            stiffness=20.0, ## 0
            damping=2.0,
            armature=0.01,
        ),
        
        "torso": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=["WJ[1-3]"],
            stiffness=20.0, ## 0
            damping=2.0,
            armature=0.01,
        )
    },
)
"""Configuration for the KIST Humanoid3 (Kimanoid) robot."""