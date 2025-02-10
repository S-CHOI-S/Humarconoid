"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-Flat-Play", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import humarconoid.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    data_log = []
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions[:,9] = 3.734696626663208
            # actions[:,10] = -4.678939342498779
            # print(f"\033[0m-------------------------------------------------------")
            # print information from the sensors
            # print(env.unwrapped.scene["contact_points"])
            # print("Received force matrix of: ", env.unwrapped.scene["contact_points"].data.force_matrix_w)
            # print("Received contact force of: ", env.unwrapped.scene["contact_points"].data.net_forces_w)
            # print(f"link_names: {env.unwrapped.scene['robot'].data.body_names}")
            # print(f"joint_names: {env.unwrapped.scene['robot'].data.joint_names}")
            
            # print(f"joint_limit[6],[9] : {env.env.scene['robot'].data.default_joint_limits[0][6]}, {env.env.scene['robot'].data.default_joint_limits[0][9]}")
            # print(f"joint_limit[7],[10]: {env.env.scene['robot'].data.default_joint_limits[0][7]}, {env.env.scene['robot'].data.default_joint_limits[0][10]}")
            # print(f"action[6],[9]:\n {actions[0][6]}, {actions[0][9]}")
            # 
            
            # print(f"obs: \n{obs[0]}")
            # print(f"action: \n{actions[0]}")
            
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            
            
            # print(f"command_norm: {torch.norm(mb_env.command_manager.get_command('base_velocity')[15, :2]).tolist()}")
            # print(f"action: {torch.norm(obs[15,:2]).tolist()}")
            
            # print(env.unwrapped.scene['robot'].data.GRAVITY_VEC_W)
            
            data_log.append({
                "timestep": timestep,
                "observation": (obs[0]).tolist(), # torch.norm(env.unwrapped.command_manager.get_command('base_velocity')[0, :2]).tolist(),  # Actions as a list
                "action": (actions[0]).tolist()   # Rewards as a list
            })
            timestep += 1
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    
    import pandas as pd
    # save data to CSV
    output_file = os.path.join(log_dir, "simulation_data.csv")
    print(f"[INFO] Saving data to {output_file}")
    df = pd.DataFrame(data_log)
    df.to_csv(output_file, index=False)
                
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
