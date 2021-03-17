import math
import numpy as np
from numpy.core.fromnumeric import mean
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


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def reward_fn(obses, acts):
    assert obses.shape[1] == 12 and acts.shape[1] == 4
    pos, ang, vel, gyr = parse_obses(obses)
    cost = np.zeros((pos.shape[0], ))
    angle = np.sum(ang[:, 0:2] ** 2, axis=1)
    cost += angle
    gyro = np.sum(gyr ** 2, axis=1)
    # gyro = np.sum(gyr ** 2, axis=1) * 0.5
    cost += gyro * 0.001
    # action = np.sum(acts ** 2, axis=1) * 0.001
    action = np.sum(acts ** 2, axis=1)
    cost += action * 0.0001
    # 高さに関して連続的な報酬を与えるように修正
    z = pos[:, 2]
    height = np.where(z < 0.3, -1. / 0.3 * z + 1.0, 0.0)
    cost += height * 0.01
    return -cost, angle, gyro, action, height
# ダイナミクスモデル


class DynamicsModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, units=(48, 48)):
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


# [課題1] ダイナミクスモデルを用いた次状態の予測
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

# ランダム方策


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


# [課題2] RSの実装
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


def parse_obses(obses):
    return obses[:, 0:3], angle_normalize(
        obses[:, 3:6]), obses[:, 6:9], angle_normalize(obses[:, 9:12])


# ダイナミクスモデルの学習用関数の定義
def fit_dynamics(n_iter=50):
    mean_loss = 0.
    for _ in range(n_iter):
        samples = dynamics_buffer.sample(batch_size)
        pos, ang, vel, gyr = parse_obses(samples["obs"])
        next_pos, next_ang, next_vel, next_gyr = parse_obses(
            samples["next_obs"])
        inputs = np.concatenate(
            [pos[:, 2, np.newaxis], ang, gyr, samples["act"]], axis=1)
        labels = np.concatenate(
            [next_pos[:, 2, np.newaxis] - pos[:, 2, np.newaxis], next_ang - ang, next_gyr - gyr], axis=1)
        mean_loss += dynamics_model.fit(
            torch.from_numpy(inputs).float(),
            torch.from_numpy(labels).float())
    return mean_loss


if __name__ == "__main__":
    episode_max_steps = 500
    if False:
        monitor_env = gym.make("takeoff-aviary-v0",
                               initial_xyzs=[[0.0, 0.0, 0.0]], gui=True)

        for episode_idx in range(5):
            monitor_env.reset()
            total_rew = 0.
            for _ in range(episode_max_steps):
                _, rew, done, _ = monitor_env.step(
                    monitor_env.action_space.sample())
                total_rew += rew
                if done:
                    break
            print("iter={0: 3d} total reward: {1: 4.4f}".format(
                episode_idx, total_rew))

    env = gym.make("takeoff-aviary-v0")

    # obs_dim = env.observation_space.high.size
    obs_dim = 7  # z, ang(3), gyr(3)
    act_dim = env.action_space.high.size

    dynamics_model = DynamicsModel(
        input_dim=obs_dim + act_dim, output_dim=obs_dim)

    # 10Kデータ分 (s, a, s') を保存できるリングバッファを用意します
    rb_dict = {
        "size": 10000,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": env.observation_space.shape},
            "next_obs": {"shape": env.observation_space.shape},
            "act": {"shape": env.action_space.shape}}}
    dynamics_buffer = ReplayBuffer(**rb_dict)

    policy = RandomPolicy(
        max_action=env.action_space.high[0],
        act_dim=env.action_space.high.size)

    # random shooting params
    n_mpc_episodes = 128
    horizon = 50

    batch_size = 500
    n_episodes = 3000

    total_steps = 0

    # ダイナミクスモデルの事前学習のために実環境でランダムに遷移を収集
    random_trial_num = 100
    for _ in range(random_trial_num):
        obs = env.reset()
        for _ in range(episode_max_steps):
            total_steps += 1
            act = env.action_space.sample()
            next_obs, _, done, _ = env.step(act)
            dynamics_buffer.add(obs=obs, act=act, next_obs=next_obs)
            obs = next_obs
            if done:
                break

    # ダイナミクスモデルの事前学習
    pre_train_dynamics_iter = 1000
    fit_dynamics(n_iter=pre_train_dynamics_iter)

    rew_list = []
    loss_list = []
    save_path = os.path.join(
        __file__, 'models', datetime.datetime.now().strftime('%m%d_%H:%M:%S'))
    os.makedirs(save_path, exist_ok=True)

    for episode_idx in range(n_episodes):
        total_rew = 0.
        total_angle_c = 0.
        total_gyro_c = 0.
        total_action_c = 0.
        total_height_c = 0.

        obs = env.reset()
        for _ in range(episode_max_steps):
            total_steps += 1

            # init obs.shape=(12, )
            assert len(obs.shape) == 1 and obs.shape[0] == 12
            # parse obses を使うためにバッチサイズ1の様に扱う
            batched_obs = obs.reshape((1, 12))

            # RSを使って1ステップだけ進める
            act = random_shooting(
                batched_obs, n_mpc_episodes=n_mpc_episodes, horizon=horizon)
            # act もバッチサイズ1の様に扱う
            assert len(act.shape) == 1 and act.shape[0] == 4
            batched_act = act.reshape((1, 4))
            next_obs, _, done, _ = env.step(act)
            rewards, angle_c, gyro_c, action_c, height_c = reward_fn(
                batched_obs, batched_act)

            # 収集した遷移をバッファに保存
            dynamics_buffer.add(obs=obs, act=act, next_obs=next_obs)

            total_rew += rewards
            total_angle_c += angle_c
            total_gyro_c += gyro_c
            total_action_c += action_c
            total_height_c += height_c
            if done:
                break
            obs = next_obs

        # ダイナミクスモデルの更新
        mean_loss = fit_dynamics(n_iter=100)
        loss_list.append(mean_loss)
        rew_list.append(total_rew)

        if episode_idx % 50 == 0 or episode_idx == 0:
            print("iter={0: 3d} total steps: {1: 5d} total reward: {2: 4.4f} mean loss: {3:.6f}".format(
                episode_idx, total_steps, total_rew[0], mean_loss))
            print("angle: {0: 4.4f} gyro: {1: 4.4f} action: {2: 4.4f} height: {3: 4.4f}".format(
                total_angle_c[0], total_gyro_c[0], total_action_c[0], total_height_c[0]))
            print("obs [angle: {}, gyr: {}, pos: {}, vel: {}]".format(
                obs[3:6], obs[9:12], obs[0:3], obs[6:9]))
        if episode_idx % 50 == 0:
            filename = str(episode_idx) + '.pt'
            torch.save({
                "random_trial_num": random_trial_num,
                "episode_max_steps": episode_max_steps,
                "n_mpc_episodes": n_mpc_episodes,
                "horizon": horizon,
                "n_episodes": n_episodes,
                "batch_size": batch_size,
                "pre_train_dynamics_iter": pre_train_dynamics_iter,
                "mean_loss": loss_list,
                "rewards": rew_list,
                "model_params": dynamics_model.model.state_dict(),
                "model_hyper_params": dynamics_model.units
            }, os.path.join(save_path, filename))
