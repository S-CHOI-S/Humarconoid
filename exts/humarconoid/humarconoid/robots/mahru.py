
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from humarconoid.robots import HUMARCONOID_EXT_DIR

MAHRU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{HUMARCONOID_EXT_DIR}/MAHRU/mahru.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "RL[1-4]+_joint",
                "LL[1-4]+_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "RL[1-4]+_joint": 150.0,
                "LL[1-4]+_joint": 150.0,
            },
            damping={
                "RL[1-4]+_joint": 5.0,
                "LL[1-4]+_joint": 5.0,
            },
            armature={
                "RL[1-4]+_joint": 0.01,
                "LL[1-4]+_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=["[RL]W_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "RA[1-4]+_joint",
                "LA[1-4]+_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["TOP_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        )
    },
)
"""Configuration for the KIST MAHRU Humanoid robot."""