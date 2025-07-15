from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.managers import RewardTermCfg as RewTerm
from humarconoid.tasks.jiwon.velocity.config.g1.rough_env_cfg import JiwonRewards

from .rough_env_cfg import JiwonRoughEnvCfg
import humarconoid.tasks.jiwon.velocity.mdp as mdp

from humarconoid.robots.g1_kist import G1_KIST_WHOLEBODY_CFG

@configclass
class JiwonWholebodyRewards(JiwonRewards):
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    
    flat_orientation_torso = RewTerm(func=mdp.flat_orientation_body, weight=-1.0)


@configclass
class JiwonWholebodyEnvCfg(JiwonRoughEnvCfg):
    rewards: JiwonWholebodyRewards = JiwonWholebodyRewards()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = G1_KIST_WHOLEBODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Action
        self.actions.joint_pos.scale = 0.5
        
        # Terrain
        self.scene.terrain.max_init_terrain_level = 0
        
        # Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.75
        self.rewards.track_ang_vel_z_exp.weight = 4.5
        
        self.rewards.dof_acc_l2.weight = -2e-06
        self.rewards.action_rate_l2.weight = -0.75
        
        self.rewards.feet_air_time.weight = 0.15

        self.rewards.joint_deviation_hip.weight = -1.25
        self.rewards.joint_deviation_knee.weight = -0.2

        self.rewards.symmetric_gait_phase.weight = 1.0
        self.rewards.symmetric_leg_phase.weight = 0.04
        
        self.rewards.joint_deviation_arms.weight = -0.3
        self.rewards.joint_deviation_waists.weight = -1.0
        
        self.rewards.alive.weight = 0.3
        
        self.rewards.energy.weight = -2.5e-07
        
        self.rewards.flat_orientation_torso.weight = -0.01
        

class JiwonWholebodyEnvCfg_PLAY(JiwonWholebodyEnvCfg):
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

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.push_robot_levels = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)