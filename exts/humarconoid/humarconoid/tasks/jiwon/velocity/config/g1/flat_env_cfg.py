from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from .rough_env_cfg import JiwonRoughEnvCfg

from humarconoid.terrains.noise import NOISE_TERRAINS_CFG

@configclass
class JiwonFlatEnvCfg(JiwonRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # self.sim.dt = 0.002 ## 500 hz
        # self.decimation = 4 ## 125 hz
        # self.actions.joint_pos.joint_names = {".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"}

        # change terrain to flat
        self.scene.terrain.terrain_generator = NOISE_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 2
        
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        
        # no terrain curriculum
        # self.curriculum.terrain_levels = None
        
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
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

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -3.0e-7
        self.rewards.feet_air_time.weight = 1.25
        self.rewards.feet_air_time.params["threshold"] = 1.5
        self.rewards.dof_torques_l2.weight = -4.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint",  ".*_ankle_.*"]
        )
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*"])
        self.rewards.feet_safe_contact.weight = -0.04
        # self.rewards.undesired_contacts.weight =-1.0
        # self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        #     "robot", body_names=[".*_hip_.*", ".*_knee_joint",  ".*_ankle_.*"]
        # )
        # self.rewards.joint_deviation_torso.weight = -0
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.0)
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
        self.events.push_robot = None
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
