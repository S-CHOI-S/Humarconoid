from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from .rough_env_cfg import JiwonRoughEnvCfg


@configclass
class JiwonArmEnvCfg(JiwonRoughEnvCfg):
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
        
        # Events
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
        self.events.push_robot.params={
            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}
            }
        
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

        # Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["^(?!.*_knee_).*"] # , ".*_knee_joint"]
        )
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 1.0
        self.rewards.feet_slide.weight = -0.2
        self.rewards.dof_torques_l2.weight = -4.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_ankle_.*", ".*_knee_.*"] # , ".*_knee_joint"]
        )
        self.rewards.feet_safe_contact.weight = -0.075
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.feet_swing_height.weight = 0.0025
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)


class JiwonArmEnvCfg_PLAY(JiwonArmEnvCfg):
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
