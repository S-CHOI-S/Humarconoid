from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm

import humarconoid.tasks.g1.velocity.mdp as mdp
from humarconoid.tasks.g1.velocity.velocity_env_cfg import CurriculumCfg

##
# Pre-defined configs
##
from humarconoid.tasks.g1.velocity.velocity_env_cfg import ARCRobotEnvCfg

@configclass
class G1WholebodyFlatCurriculum(CurriculumCfg):
    push_robot_levels = CurrTerm(func=mdp.push_robot_levels)


@configclass
class G1WholebodyFlatEnvCfg(ARCRobotEnvCfg):
    curriculum: G1WholebodyFlatCurriculum = G1WholebodyFlatCurriculum()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # change episode length
        self.episode_length_s = 250.0
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # Curriculums
        self.curriculum.terrain_levels = None
        
        # Rewards
        self.rewards.track_lin_vel_xy.weight = 1.25
        self.rewards.track_ang_vel_z.weight = 0.75
        
        self.rewards.alive.weight = 0.25
        
        self.rewards.base_linear_velocity.weight = -2.0
        self.rewards.base_angular_velocity.weight = -0.05
        
        self.rewards.joint_vel.weight = -0.001
        self.rewards.joint_acc.weight = -2.5e-07
        
        self.rewards.action_rate.weight = -0.05
        
        self.rewards.dof_pos_limits.weight = -5.0
        
        self.rewards.energy.weight = -1e-05
        
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -1.0
        
        self.rewards.flat_orientation_l2.weight = -5.0
        
        self.rewards.base_height.weight = -10
        
        self.rewards.gait.weight = 0.5
        
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_clearance.weight = 1.0
        
        self.rewards.undesired_contacts.weight = -1.0


@configclass
class G1WholebodyFlatEnvCfg_PLAY(G1WholebodyFlatEnvCfg):
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
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = False
        
        # Commands
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
