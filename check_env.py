import os
import torch
import mujoco
import gymnasium as gym
import gym_aloha
import lerobot

print("=" * 60)
print("1. PyTorch & GPU Check")
print(f"   PyTorch version : {torch.__version__}")
print(f"   CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device      : {torch.cuda.get_device_name(0)}")

print("\n2. MuJoCo & LeRobot Check")
print(f"   MuJoCo version  : {mujoco.__version__}")
print(f"   LeRobot version : {lerobot.__version__}")
print(f"   MUJOCO_GL       : {os.environ.get('MUJOCO_GL', 'default')}")

print("\n3. Creating ALOHA Bimanual Robot Environment (AlohaTransferCube-v0)...")
env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
obs, info = env.reset()
print(f"   Camera image shape (top) : {obs['pixels']['top'].shape}")
print(f"   Robot joint state shape  : {obs['agent_pos'].shape} (14-DoF: 6 joints + 1 gripper per arm)")
print(f"   Action space             : {env.action_space.shape}")
env.close()
print("=" * 60)
print("Environment check passed successfully!")
