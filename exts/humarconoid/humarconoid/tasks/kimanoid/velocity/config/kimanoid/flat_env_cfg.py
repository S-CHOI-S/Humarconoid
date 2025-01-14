from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from .rough_env_cfg import KimanoidRoughEnvCfg


@configclass
class KimanoidFlatEnvCfg(KimanoidRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 1.5 # 0.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*LJ[1-7]"]
        )
        
        
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
        self.rewards.heel_toe_air_time.weight = 1.0
        self.rewards.reference_motion.weight = 0.25
        self.rewards.contact_motion.weight = 1.0
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class KimanoidFlatEnvCfg_PLAY(KimanoidFlatEnvCfg):
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
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
