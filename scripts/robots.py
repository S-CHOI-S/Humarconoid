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

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Simple simulation for HumARConoid robots.")

parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# Import extensions to set up environment tasks
import humarconoid.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
    
def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # create isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
        
    # Play the simulator
    print("[INFO]: Setup complete...")
    
    # Simulate physics
    count = 0
    prev_obs = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 2000 == 0:
                count = 0
                env.reset()
                
            # apply actions to the robot
            # efforts = torch.rand_like(env.action_manager.action) * 2
            efforts = torch.zeros_like(env.action_manager.action) * 2
            efforts[0][6] = 0.5
            efforts[0][7] = -0.5
            
            efforts[0][9] = 1.0
            efforts[0][10] = -1.0
            
            efforts[0][11] = 0.5
            efforts[0][12] = -0.5
            
            # print("Shape of asset.data.root_state_w:", env.scene["robot"].root_state_w.shape)

            distance = torch.linalg.norm(env.scene["robot"].data.body_state_w[:, 14] - env.scene["robot"].data.body_state_w[:, 15])
            print(env.scene["robot"].data.joint_names)

            if distance < 0.2:
                distance = torch.tensor(0.0)
            print(distance)

            obs, rew, terminated, truncated, info = env.step(efforts)
            # print("=============================================")
            # print(efforts[0][9])
            
            count += 1
            
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()