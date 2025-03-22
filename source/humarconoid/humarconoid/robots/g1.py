import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from humarconoid.robots import HUMARCONOID_EXT_DIR

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{HUMARCONOID_EXT_DIR}/G1/g1.usd",
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
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""

'''
* Body names:
    ['pelvis', 'left_hip_pitch_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 
     'head_link', 'imu_link', 'left_shoulder_pitch_link', 'logo_link', 'right_shoulder_pitch_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 
     'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 
     'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_elbow_pitch_link', 'right_elbow_pitch_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 
     'left_elbow_roll_link', 'right_elbow_roll_link', 'left_palm_link', 'right_palm_link', 'left_five_link', 'left_three_link', 'left_zero_link', 
     'right_five_link', 'right_three_link', 'right_zero_link', 'left_six_link', 'left_four_link', 'left_one_link', 'right_six_link', 
     'right_four_link', 'right_one_link', 'left_two_link', 'right_two_link']

* Joint names:
    [             '0',                            '1',                         '2',                         '3',                        '4',   
    [     'left_hip_pitch_joint',       'right_hip_pitch_joint',              'torso_joint',      'left_hip_roll_joint',       'right_hip_roll_joint', 
 
                  '5',                            '6',                         '7',                         '8',                        '9', 
     'left_shoulder_pitch_joint',  'right_shoulder_pitch_joint',       'left_hip_yaw_joint',      'right_hip_yaw_joint',   'left_shoulder_roll_joint', 
     
                 '10',                           '11',                        '12',                        '13',                       '14',  
     'right_shoulder_roll_joint',             'left_knee_joint',         'right_knee_joint',  'left_shoulder_yaw_joint',   'right_shoulder_yaw_joint', 
     
                 '15',                           '16',                        '17',                        '18',                       '19',
        'left_ankle_pitch_joint',     'right_ankle_pitch_joint',   'left_elbow_pitch_joint',  'right_elbow_pitch_joint',      'left_ankle_roll_joint', 

                 '20',                           '21',                        '22',                        '23',                       '24',
        'right_ankle_roll_joint',       'left_elbow_roll_joint',   'right_elbow_roll_joint',          'left_five_joint',           'left_three_joint', 
        
                 '25',                           '26',                        '27',                        '28',                       '29',
               'left_zero_joint',            'right_five_joint',        'right_three_joint',         'right_zero_joint',             'left_six_joint', 

                 '30',                           '31',                        '32',                        '33',                       '34',
               'left_four_joint',              'left_one_joint',          'right_six_joint',         'right_four_joint',            'right_one_joint', 
               
                 '35',                           '36'  ]
                'left_two_joint',             'right_two_joint']

'''