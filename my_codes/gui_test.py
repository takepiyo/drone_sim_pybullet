import torch
import numpy as np
import gym
import gnwrapper
from torch import nn

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

env = gym.make("takeoff-aviary-v0")
monitor_env = gnwrapper.Monitor(gym.make("takeoff-aviary-v0"), size=(
    400, 300), directory='.', force=True, video_callable=lambda ep: True)

episode_max_steps = 300


for episode_idx in range(10):
    monitor_env.reset()
    total_rew = 0.
    for _ in range(episode_max_steps):
        _, rew, done, _ = monitor_env.step(monitor_env.action_space.sample())
        total_rew += rew
        if done:
            break
    print("iter={0: 3d} total reward: {1: 4.4f}".format(
        episode_idx, total_rew))
monitor_env.display()
env.close()
monitor_env.close()
