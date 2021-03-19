import datetime
import os
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.control.Reference2Rpm import Reference_to_Rpm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3 import A2C
from torch import nn
import torch
from cpprb import ReplayBuffer
import gnwrapper
import gym
from scipy.signal import lfilter
from numpy.core.fromnumeric import mean
import numpy as np
import math
import sys
# sys.path.append("/content/drive/MyDrive/Colab Notebooks/my_modules/")


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def reward_fn(obses, acts, episode_max_steps):
    assert obses.shape[1] == 12 and acts.shape[1] == 4
    pos, ang, vel, gyr = parse_obses(obses)
    cost = np.zeros((pos.shape[0], ))
    angle = np.sum(ang[:, 0:2] ** 2, axis=1)
    # cost += angle
    gyro = np.clip(np.sum(gyr ** 2, axis=1) / 3.0, 0.0, 1.0)
    # gyro = np.sum(gyr ** 2, axis=1) * 0.5
    cost += gyro
    # action = np.sum(acts ** 2, axis=1) * 0.001
    action = np.sum(acts ** 2, axis=1)
    # cost += action * 0.01
    # 高さに関して連続的な報酬を与えるように修正
    z = pos[:, 2]
    height = np.where(z < 0.3, -1. / 0.3 * z + 1.0, 0.0)
    # cost += height * 1.0
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
def predict_next_state(obses, acts, idx=None):
    if len(obses.shape) < 2:
        assert obses.shape[0] == 12
        obses = obses.reshape(1, 12)
    assert obses.shape[1] == 12
    if len(acts.shape) < 2:
        assert acts.shape[0] == 4
        acts = acts.reshape(1, 4)
    assert acts.shape[1] == 4

    assert obses.shape[0] == acts.shape[0] and obses.shape[1] == 12
    pos, ang, vel, gyr = parse_obses(obses)
    # ダイナミクスモデルへの入力は角速度と行動のConcata
    inputs = np.concatenate([gyr, acts], axis=1)
    inputs = torch.from_numpy(inputs).float()

    # ダイナミクスモデルの出力は次の状態と現在の状態との差分
    idx = np.random.randint(n_dynamics_model) if idx is None else idx
    obs_diffs = dynamics_models[idx].predict(inputs).data.numpy()
    batch = obs_diffs.shape[0]
    obs_diffs = np.concatenate([np.zeros((batch, 9)), obs_diffs], axis=1)
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


def parse_obses(obses):
    return obses[:, 0:3], obses[:, 3:6], obses[:, 6:9], obses[:, 9:12]


# ダイナミクスモデルの学習用関数の定義
def fit_dynamics(n_iter=50):
    mean_losses = np.zeros(shape=(n_dynamics_model,), dtype=np.float32)
    for _ in range(n_iter):
        samples = dynamics_buffer.sample(batch_size)
        pos, ang, vel, gyr = parse_obses(samples["obs"])
        next_pos, next_ang, next_vel, next_gyr = parse_obses(
            samples["next_obs"])
        inputs = np.concatenate(
            [gyr, samples["act"]], axis=1)
        labels = next_gyr - gyr
        for i, dynamics_model in enumerate(dynamics_models):
            mean_losses[i] += dynamics_model.fit(
                torch.from_numpy(inputs).float(),
                torch.from_numpy(labels).float())
    return mean_losses

# PPO functions


def calculate_log_pi(log_stds, noises, actions):
    return (
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True)
        - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
        - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True))


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)

    actions = means + noises * log_stds.exp()
    actions = torch.tanh(actions)

    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def compute_log_probs(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def discount_cumsum(x, discount):
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0)[::-1]


class GaussianActor(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

        self.device = device

    def forward(self, states):
        states = self._format_shape(states)

        return torch.tanh(self.net(states))

    def sample(self, states):
        states = self._format_shape(states)
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        states = self._format_shape(states)
        return compute_log_probs(self.net(states), self.log_stds, actions)

    def _format_shape(self, states):
        if len(states.shape) < 2:
            assert states.shape[0] == 12
            states = states.reshape(1, 12)
        assert states.shape[1] == 12
        pos, ang, vel, gyr = parse_obses(states)
        # states = np.concatenate([gyr], axis=1)
        states = gyr
        assert states.shape[1] == 3
        states = torch.tensor(states, dtype=torch.float,
                              device=self.device)
        return states


class Critic(nn.Module):
    def __init__(self, state_shape, device):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.device = device

    def forward(self, states):
        states = self._format_shape(states)
        return self.net(states)

    def _format_shape(self, states):
        if len(states.shape) < 2:
            assert states.shape[0] == 12
            states = states.reshape(1, 12)
        assert states.shape[1] == 12
        pos, ang, vel, gyr = parse_obses(states)
        # states = np.concatenate([pos[:, 2, np.newaxis], ang, gyr], axis=1)
        states = gyr
        assert states.shape[1] == 3
        states = torch.tensor(states, dtype=torch.float,
                              device=self.device)
        return states


class PPO:
    def __init__(self,
                 state_shape,
                 action_shape,
                 mixing_obj,
                 max_action=1.,
                 device=torch.device('cpu'),
                 seed=0,
                 batch_size=64,
                 lr=3e-4,
                 discount=0.9,
                 horizon=2048,
                 n_epoch=10,
                 clip_eps=0.2,
                 lam=0.95,
                 coef_ent=0.,
                 max_grad_norm=10.):
        fix_seed(seed)

        self.actor = GaussianActor(
            state_shape, action_shape, device).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_shape, device).to(device)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.horizon = horizon
        self.n_epoch = n_epoch
        self.clip_eps = clip_eps
        self.lam = lam
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

        self.mixing = mixing_obj

    def get_action(self, state, test=False):
        with torch.no_grad():
            if test:
                action = self.actor(state)
            else:
                action, _ = self.actor.sample(state)
        rpm = self.mixing.get_rpm(action.cpu().numpy() * self.max_action)
        return rpm

    def get_action_and_val(self, state):
        with torch.no_grad():
            action, logp = self.actor.sample(state)
            value = self.critic(state)
            action = self.mixing.get_rpm(
                action.to('cpu').detach().numpy() * self.max_action)
        return action, logp, value

    def train(self, states, actions, advantages, logp_olds, returns):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions / self.max_action).float()
        advantages = torch.from_numpy(advantages).float()
        logp_olds = torch.from_numpy(logp_olds).float()
        returns = torch.from_numpy(returns).float()
        self.update_actor(states, actions, logp_olds, advantages)
        self.update_critic(states, returns)

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, logp_olds, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - logp_olds).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean(
        ) - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

# 複数のエピソードで現在の方策を評価し平均リターンを返す


def evaluate_policy(total_steps, test_episodes=10):
    avg_test_return = 0.
    avg_total_angle_c = 0.
    avg_total_gyro_c = 0.
    avg_total_action_c = 0.
    avg_total_height_c = 0.
    for i in range(test_episodes):
        episode_return = 0.
        total_angle_c = 0.
        total_gyro_c = 0.
        total_action_c = 0.
        total_height_c = 0.
        obs = env.reset()
        for _ in range(episode_max_steps):
            act = policy.get_action(obs, test=True)
            next_obs, _, _, _ = env.step(act)
            rews = reward_fn(obs.reshape((1, 12)),
                             act.reshape((1, 4)), episode_max_steps)
            episode_return += rews[0]
            total_angle_c += rews[1]
            total_gyro_c += rews[2]
            total_action_c += rews[3]
            total_height_c += rews[4]
            obs = next_obs
        avg_test_return += episode_return / test_episodes
        avg_total_angle_c += total_angle_c / test_episodes
        avg_total_gyro_c += total_gyro_c / test_episodes
        avg_total_action_c += total_action_c / test_episodes
        avg_total_height_c += total_height_c / test_episodes
    return avg_test_return, avg_total_angle_c, avg_total_gyro_c, avg_total_action_c, avg_total_height_c


def collect_transitions_real_env():
    obs = env.reset()
    episode_steps = 0
    for _ in range(policy.horizon):
        episode_steps += 1
        act = policy.get_action(obs)
        # 実環境でロールアウト
        next_obs, *_ = env.step(act)
        dynamics_buffer.add(obs=obs, act=act, next_obs=next_obs)
        obs = next_obs
        if episode_steps == episode_max_steps:
            episode_steps = 0
            obs = env.reset()

# ダイナミクスモデルを用いた方策学習用サンプルの生成


def collect_transitions_sim_env():
    on_policy_buffer.clear()
    n_episodes = 0
    ave_episode_return = 0
    while on_policy_buffer.get_stored_size() < policy.horizon:
        # 実環境で初期値を取得
        obs = env.reset().reshape(1, 12)
        episode_return = 0.
        for i in range(episode_max_steps):
            act, logp, val = policy.get_action_and_val(obs)
            # ダイナミクスモデルを用いて次状態を予測
            next_obs = predict_next_state(obs, act)
            rew = reward_fn(obs, act, episode_max_steps)[0]
            episode_buffer.add(obs=obs, act=act, next_obs=next_obs, rew=rew,
                               done=False, logp=logp, val=val)
            obs = next_obs
            episode_return += rew
        finish_horizon(last_val=val)
        ave_episode_return += episode_return
        n_episodes += 1
    return ave_episode_return / n_episodes


# PPOの学習のため、エピソード終了時に必要な計算
def finish_horizon(last_val=0):
    samples = episode_buffer.get_all_transitions()
    rews = np.append(samples["rew"], last_val)
    vals = np.append(samples["val"], last_val)

    # GAE-Lambda
    deltas = rews[:-1] + policy.discount * vals[1:] - vals[:-1]
    advs = discount_cumsum(deltas, policy.discount * policy.lam)

    # 価値関数学習の際のターゲットとなるリターンを計算
    rets = discount_cumsum(rews, policy.discount)[:-1]
    on_policy_buffer.add(
        obs=samples["obs"], act=samples["act"], done=samples["done"],
        ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
    episode_buffer.clear()


def update_policy():

    # 前準備としてAdvantageの平均と分散を計算
    samples = on_policy_buffer.get_all_transitions()
    mean_adv = np.mean(samples["adv"])
    std_adv = np.std(samples["adv"])

    for _ in range(policy.n_epoch):
        samples = on_policy_buffer._encode_sample(
            np.random.permutation(policy.horizon))
        adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
        actor_loss, critic_loss = 0., 0.
        for idx in range(int(policy.horizon / policy.batch_size)):
            target = slice(idx * policy.batch_size,
                           (idx + 1) * policy.batch_size)
            policy.train(
                states=samples["obs"][target],
                actions=samples["act"][target],
                advantages=adv[target],
                logp_olds=samples["logp"][target],
                returns=samples["ret"][target])


def evaluate_current_return(init_states):
    # 同じ初期値で評価できるように、関数内で初期値を生成せず引数として与える
    n_episodes = n_dynamics_model * n_eval_episodes_per_model
    assert init_states.shape[0] == n_episodes

    obses = init_states.copy()
    next_obses = np.zeros_like(obses)
    returns = np.zeros(shape=(n_episodes,), dtype=np.float32)

    for _ in range(episode_max_steps):
        # 現在の方策を用いて行動を生成
        acts = policy.get_action(obses, test=True)
        for i in range(n_episodes):
            model_idx = i // n_eval_episodes_per_model
            env_act = np.clip(
                acts[i], env.action_space.low, env.action_space.high)
            next_obses[i] = predict_next_state(
                obses[i], env_act, idx=model_idx)
        returns += reward_fn(obses, acts, episode_max_steps)[0]
        obses = next_obses

    return returns


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

    env = gym.make("takeoff-aviary-v0", initial_xyzs=[[0.0, 0.0, 0.0]])
    ref_to_rpm = Reference_to_Rpm()

    # obs_dim = env.observation_space.high.size
    obs_dim = 3  # gyr(3)
    act_dim = env.action_space.high.size
    torques_dim = 3

    n_dynamics_model = 5
    n_eval_episodes_per_model = 5

    dynamics_models = [DynamicsModel(
        input_dim=obs_dim + act_dim, output_dim=obs_dim) for _ in range(n_dynamics_model)]

    policy = PPO((obs_dim, ),
                 (torques_dim, ),
                 mixing_obj=ref_to_rpm,
                 max_action=3200)

    # 10Kデータ分 (s, a, s') を保存できるリングバッファを用意します
    rb_dict = {
        "size": 10000,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": env.observation_space.shape},
            "next_obs": {"shape": env.observation_space.shape},
            "act": {"shape": env.action_space.shape}}}
    dynamics_buffer = ReplayBuffer(**rb_dict)
    rb_dict = {
        "size": policy.horizon,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": env.observation_space.shape},
            "act": {"shape": env.action_space.shape},
            "done": {},
            "logp": {},
            "ret": {},
            "adv": {}}}
    on_policy_buffer = ReplayBuffer(**rb_dict)
    rb_dict = {
        "size": episode_max_steps,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": env.observation_space.shape},
            "act": {"shape": env.action_space.shape},
            "next_obs": {"shape": env.observation_space.shape},
            "rew": {},
            "done": {},
            "logp": {},
            "val": {}}}
    episode_buffer = ReplayBuffer(**rb_dict)

    total_steps = 0
    test_episodes = 10

    batch_size = 500

    rew_list = []
    total_step_list = []
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'models', datetime.datetime.now().strftime('%m%d_%H:%M:%S'))
    os.makedirs(save_path, exist_ok=True)

    while True:
        # 実環境でダイナミクスモデルを学習するためのサンプルを収集
        collect_transitions_real_env()
        total_steps += policy.horizon

        # ダイナミクスモデルの学習
        fit_dynamics()

        n_updates = 0
        # 方策評価のための初期値の生成
        init_states_for_eval = np.array([
            env.reset() for _ in range(n_dynamics_model * n_eval_episodes_per_model)])

        # 方策更新前の性能評価
        returns_before_update = evaluate_current_return(init_states_for_eval)
        while True:
            n_updates += 1

            # ダイナミクスモデルを用いて方策学習用のサンプルを生成
            average_return = collect_transitions_sim_env()

            # 方策更新
            update_policy()

            # 方策更新後の性能評価
            returns_after_update = evaluate_current_return(
                init_states_for_eval)

            # 方策更新による性能評価の割合を計算
            improved_ratio = np.sum(returns_after_update > returns_before_update) / (
                n_dynamics_model * n_eval_episodes_per_model)

            # 方策更新による性能向上があまり見られない場合、ループを抜ける
            if improved_ratio < 0.7:
                print("Training total steps: {0: 7} improved ratio: {1: .2f} simulated return: {2: .4f} n_update: {3: 2}".format(
                    total_steps, improved_ratio, average_return[0], n_updates))
                break
            returns_before_update = returns_after_update.copy()

        # 実環境での方策評価
        if total_steps // policy.horizon % 10 == 0:
            rew_and_costs = evaluate_policy(total_steps, test_episodes)
            print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                total_steps, rew_and_costs[0][0], test_episodes))
            print("angle: {0: 4.4f} gyro: {1: 4.4f} action: {2: 4.4f} height: {3: 4.4f}".format(
                rew_and_costs[1][0], rew_and_costs[2][0], rew_and_costs[3][0], rew_and_costs[4][0]))
            rew_list.append(rew_and_costs[0])
            total_step_list.append(total_steps)
            filename = str(total_steps) + '.pt'
            torch.save({
                "episode_max_steps": episode_max_steps,
                "n_dynamics_model": n_dynamics_model,
                "rewards": rew_list,
                "steps": total_step_list,
                "model_actor_params": policy.actor.net.state_dict(),
                "model_critic_params": policy.critic.net.state_dict(),
            }, os.path.join(save_path, filename))

        # 時間がかかるので50回で終了とする
        # if total_steps // policy.horizon % 50 == 0:
        #     break
