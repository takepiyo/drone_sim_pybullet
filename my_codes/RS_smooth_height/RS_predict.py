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

from RS_train import DynamicsModel, reward_fn, RandomPolicy, parse_obses


def random_shooting(init_obs, n_mpc_episodes=128, horizon=50):
    init_actions = policy.get_actions(batch_size=n_mpc_episodes)
    returns = np.zeros(shape=(n_mpc_episodes,))
    obses = np.tile(init_obs, (n_mpc_episodes, 1))
    # obses = np.tile(np.concatenate([ang, gyr], axis=1), (n_mpc_episodes, 1))

    # horizon分未来まで予測
    for i in range(horizon):
        # 行動の生成。最初のステップ目の時のみ上記 init_actions を使う（上書きしない）
        acts = init_actions if i == 0 else policy.get_actions(
            batch_size=n_mpc_episodes)

        # ダイナミクスモデルを用いた次状態の予測
        next_obses = predict_next_state(obses, acts)
        # next_obses は z ang gyr しか意味を持ってない(モデルはx y vel を予測しない)

        # 報酬は事前に定義した報酬関数を用いる
        rewards, _, _, _, _ = reward_fn(obses, acts)
        returns += rewards
        obses = next_obses

    # 最も累積報酬が高かった最初の行動を返す
    return init_actions[np.argmax(returns)]


def predict_next_state(obses, acts):
    assert obses.shape[0] == acts.shape[0] and obses.shape[1] == 12
    pos, ang, vel, gyr = parse_obses(obses)
    # ダイナミクスモデルへの入力は状態と行動の Concata
    inputs = np.concatenate([pos[:, 2, np.newaxis], ang, gyr, acts], axis=1)
    inputs = torch.from_numpy(inputs).float()

    # ダイナミクスモデルの出力は次の状態と現在の状態との差分
    obs_diffs = dynamics_model.predict(inputs).data.numpy()
    batch = obs_diffs.shape[0]
    obs_diffs = np.concatenate([np.zeros((batch, 2)), obs_diffs[:, 0:4],  # z roll pitch yaw
                                np.zeros((batch, 3)), obs_diffs[:, 4:7]], axis=1)
    assert obses.shape == obs_diffs.shape
    next_obses = obses + obs_diffs
    return next_obses


if __name__ == "__main__":
    checkpoint = torch.load(
        "/home/takeshi/gym-pybullet-drones/my_codes/RS_sparse_height/models/colab_0317_19:55/1150.pt")
    # dynamics_model = checkpoint["model_object"]
    dynamics_model = DynamicsModel(11, 7, checkpoint["model_hyper_params"])
    dynamics_model.model.load_state_dict(checkpoint["model_params"])
    episode_max_steps = checkpoint["episode_max_steps"]

    env = gym.make("takeoff-aviary-v0",
                   initial_xyzs=[[0.0, 0.0, 0.0]], gui=True)

    num_trial = 5
    policy = RandomPolicy(
        max_action=env.action_space.high[0],
        act_dim=env.action_space.high.size)

    for episode_idx in range(num_trial):
        total_rew = 0.
        total_angle_c = 0.
        total_gyro_c = 0.
        total_action_c = 0.
        total_height_c = 0.

        obs = env.reset()
        logger = Logger(logging_freq_hz=env.SIM_FREQ, num_drones=1)
        for i in range(episode_max_steps):

            # init obs.shape=(12, )
            assert len(obs.shape) == 1 and obs.shape[0] == 12
            # parse obses を使うためにバッチサイズ1の様に扱う
            batched_obs = obs.reshape((1, 12))

            # RSを使って1ステップだけ進める
            act = random_shooting(batched_obs)
            # act もバッチサイズ1の様に扱う
            assert len(act.shape) == 1 and act.shape[0] == 4
            batched_act = act.reshape((1, 4))
            next_obs, _, done, _ = env.step(act)
            logger.log(drone=0,
                       timestamp=i/env.SIM_FREQ,
                       state=np.hstack(
                           [obs[0:3], obs[6:9], obs[3:6], obs[9:12], np.resize(act, (4))]),
                       control=np.zeros(12))
            rewards, angle_c, gyro_c, action_c, height_c = reward_fn(
                batched_obs, batched_act)
            total_rew += rewards
            total_angle_c += angle_c
            total_gyro_c += gyro_c
            total_action_c += action_c
            total_height_c += height_c
            if done:
                break
            obs = next_obs
        print("total reward: {}".format(total_rew[0]))
        print("angle: {0: 4.4f} gyro: {1: 4.4f} action: {2: 4.4f} height: {3: 4.4f}".format(
            total_angle_c[0], total_gyro_c[0], total_action_c[0], total_height_c[0]))
        logger.save()
        logger.plot()
