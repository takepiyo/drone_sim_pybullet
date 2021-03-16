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


class DynamicsModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, units=(32, 32)):
        super().__init__()

        # 隠れ層2層のMLP
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, units[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(units[0], units[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(units[1], output_dim)
        )

        self.units = units
        self._loss_fn = torch.nn.MSELoss(reduction='mean')
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, inputs):
        assert inputs.ndim == 2
        return self.model(inputs)

    def fit(self, inputs, labels):
        predicts = self.predict(inputs)
        loss = self._loss_fn(predicts, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.data.numpy()


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def parse_obses(obses):
    return obses[:, 0:3], angle_normalize(
        obses[:, 3:6]), obses[:, 6:9], angle_normalize(obses[:, 9:12])


def reward_fn(obses, acts):
    pos, ang, vel, gyr = parse_obses(obses)
    cost = np.zeros((pos.shape[0], ))
    cost += np.sum(ang[:, 0:2] ** 2, axis=1)
    cost += np.sum(gyr ** 2, axis=1) * 0.1
    cost += np.sum(acts ** 2, axis=1) * 0.001
    cost += (pos[:, 2] < 0.1).astype(np.float32)
    return -cost


class RandomPolicy:
    def __init__(self, max_action, act_dim):
        self._max_action = max_action  # action の最大値
        self._act_dim = act_dim  # action の次元数

    def get_actions(self, batch_size):
        # 一様分布からバッチサイズ分ランダムにサンプリング
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=(batch_size, self._act_dim))


def random_shooting(init_obs, n_mpc_episodes=64, horizon=20):
    init_actions = policy.get_actions(batch_size=n_mpc_episodes)
    pos, ang, vel, gyr = parse_obses(init_obs)

    returns = np.zeros(shape=(n_mpc_episodes,))
    # obses = np.tile(init_obs, (n_mpc_episodes, 1))
    obses = np.tile(np.concatenate([ang, gyr], axis=1), (n_mpc_episodes, 1))

    # horizon分未来まで予測
    for i in range(horizon):
        # 行動の生成。最初のステップ目の時のみ上記 init_actions を使う（上書きしない）
        acts = init_actions if i == 0 else policy.get_actions(
            batch_size=n_mpc_episodes)

        # ダイナミクスモデルを用いた次状態の予測
        next_obses = predict_next_state(obses, acts)

        # 報酬は事前に定義した報酬関数を用いる
        rewards = reward_fn(obses, acts)
        returns += rewards
        obses = next_obses

    # 最も累積報酬が高かった最初の行動を返す
    return init_actions[np.argmax(returns)]

# [課題1] ダイナミクスモデルを用いた次状態の予測


def predict_next_state(obses, acts):
    assert obses.shape[0] == acts.shape[0]
    # ダイナミクスモデルへの入力は状態と行動の Concata
    inputs = np.concatenate([obses, acts], axis=1)
    inputs = torch.from_numpy(inputs).float()

    # ダイナミクスモデルの出力は次の状態と現在の状態との差分
    obs_diffs = dynamics_model.predict(inputs).data.numpy()
    assert obses.shape == obs_diffs.shape
    next_obses = obses + obs_diffs
    return next_obses


if __name__ == "__main__":
    checkpoint = torch.load(
        "/home/takeshi/gym-pybullet-drones/my_codes/models/0316_19:21:55.pt")
    # dynamics_model = checkpoint["model_object"]
    dynamics_model = DynamicsModel(10, 6, checkpoint["model_hyper_params"])
    dynamics_model.model.load_state_dict(checkpoint["model_params"])
    episode_max_steps = checkpoint["episode_max_steps"]

    env = gym.make("takeoff-aviary-v0", gui=True)

    num_trial = 5
    policy = RandomPolicy(
        max_action=env.action_space.high[0],
        act_dim=env.action_space.high.size)

    for episode_idx in range(num_trial):
        total_rew = 0.

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
            rewards = reward_fn(batched_obs, batched_act)
            total_rew += rewards
            if done:
                break
            obs = next_obs
        print("total reward: {}".format(total_rew[0]))
        logger.save()
        logger.plot()
