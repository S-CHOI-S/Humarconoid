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
        # usd_path=f"{HUMARCONOID_EXT_DIR}/KIST_HUMANOID_TORSO/kist_humanoid3.usd",
        usd_path=f"{HUMARCONOID_EXT_DIR}/KIST_HUMANOID_TORSO_ANKLE/kist_humanoid3.usd",
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
            # "LLJ1": -0.0624,
            # "LLJ2": -0.0311,
            # "LLJ3": 0.421,
            # "LLJ4": 0.95,
            # "LLJ5": 0.52,
            # "LLJ6": 0.0471,
            # "LLJ7": 0,
            # "RLJ1": 0.047,
            # "RLJ2": 0.0131,
            # "RLJ3": -0.421,
            # "RLJ4": -0.95,
            # "RLJ5": -0.52,
            # "RLJ6": -0.055,
            # "RLJ7": 0,
            # "BWJ1": 0.00103,
            # "BWJ2": -0.00247,
            # "BWJ3": 0.532,
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
            effort_limit={
                ".*LJ[1-2]": 100,
                ".*LJ[3-4]": 300,
            },
            velocity_limit={
                ".*LJ[1-2]": 192,
                ".*LJ[3-4]": 93,
            },
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
            joint_names_expr=["LLJ[5-7]", "RLJ[5-7]"],
            effort_limit={
                ".*LJ[5-6]": 100,
                ".*LJ7": 20, # 10 
            },
            velocity_limit={
                ".*LJ[5-6]": 192,
                ".*LJ7": 144,
            },
            stiffness=20.0, ## 0
            damping=2.0,
            armature=0.01,
        ),
        
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["BWJ[1-3]"],
            effort_limit=300,
            velocity_limit=93,
            stiffness=20.0, ## 0
            damping=2.0,
            armature=0.01,
        )
    },
)
"""Configuration for the KIST Humanoid3 (Kimanoid) robot."""

'''
* Body names:
    ['kimanoid',     'Left_Leg1',    'Right_Leg1',   'Body_Waist1',    'Left_Leg2',   'Right_Leg2',   'Body_Waist2', 
     'Left_Leg3',   'Right_Leg3',   'Body_Waist3',     'Left_Leg4',   'Right_Leg4',    'Left_Leg5',    'Right_Leg5', 
     'Left_Leg6',   'Right_Leg6',     'Left_Leg7',    'Right_Leg7']

* Joint names:
    [   '0',    '1',    '2',    '3',    '4',    '5',    '6',    '7',    '8',    '9',   '10',   '11',   '12',   '13',   '14',   '15',   '16']
    ['LLJ1', 'RLJ1', 'BWJ1', 'LLJ2', 'RLJ2', 'BWJ2', 'LLJ3', 'RLJ3', 'BWJ3', 'LLJ4', 'RLJ4', 'LLJ5', 'RLJ5', 'LLJ6', 'RLJ6', 'LLJ7', 'RLJ7']

'''
