import math
import numpy as np
from scipy.signal import lfilter
import gym
import gnwrapper
from cpprb import ReplayBuffer
import torch
from torch import nn

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

import os
import datetime

from train import PPO, reward_fn

if __name__ == "__main__":
    env = gym.make("takeoff-aviary-v0",
                   initial_xyzs=[[0.0, 0.0, 0.0]], gui=True)
    checkpoint = torch.load(
        "my_codes/En_PPO/models/0318_05_12_44/1515520.pt")
    obs_dim = 7  # z, ang(3), gyr(3)
    act_dim = env.action_space.high.size

    policy = PPO((obs_dim, ),
                 (act_dim, ),
                 max_action=env.action_space.high[0])

    policy.actor.net.load_state_dict(checkpoint["model_actor_params"])
    policy.critic.net.load_state_dict(checkpoint["model_critic_params"])

    episode_max_steps = checkpoint["episode_max_steps"]

    episode_return = 0.
    total_angle_c = 0.
    total_gyro_c = 0.
    total_action_c = 0.
    total_height_c = 0.
    obs = env.reset()
    logger = Logger(logging_freq_hz=env.SIM_FREQ, num_drones=1)
    for i in range(episode_max_steps):
        act = policy.get_action(obs, test=True)
        next_obs, _, _, _ = env.step(act)
        logger.log(drone=0,
                   timestamp=i / env.SIM_FREQ,
                   state=np.hstack(
                       [obs[0:3], obs[6:9], obs[3:6], obs[9:12], np.resize(act, (4))]),
                   control=np.zeros(12))
        obs = next_obs

        rew, angle_c, gyro_c, action_c, height_c = reward_fn(
            obs.reshape((1, 12)), act.reshape((1, 4)), episode_max_steps)
        episode_return += rew
        total_angle_c += angle_c
        total_gyro_c += gyro_c
        total_action_c += action_c
        total_height_c += height_c
    print("reward; ", episode_return[0])
    print("angle: {0: 4.4f} gyro: {1: 4.4f} action: {2: 4.4f} height: {3: 4.4f}".format(
        total_angle_c[0], total_gyro_c[0], total_action_c[0], total_height_c[0]))
    logger.save()
    logger.plot()
