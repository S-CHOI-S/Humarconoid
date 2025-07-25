from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from humarconoid.robots import KAPEX_CFG
from humarconoid.tasks.kapex.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
import humarconoid.tasks.kapex.velocity.mdp as mdp


@configclass
class KapexStandRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Leg6"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Leg6"),
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[1-2]"])},
    )

    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.00,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*LJ[5-6]"])},
    )


@configclass
class KapexStandEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KapexStandRewardsCfg = KapexStandRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = KAPEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Observations (Policy)
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        # Observations (Critic)
        self.observations.critic.base_lin_vel = None
        self.observations.critic.height_scan = None

        # Events
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -1.0

        self.rewards.termination_penalty.weight = -200.0
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.joint_deviation_ankle.weight = -0.5

        # Terminations
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names = ["^(?!.*_Leg[6-7]$).*"]), "threshold": 1.0
        }

        # Curriculums

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


class KapexStandEnvCfg_PLAY(KapexStandEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
