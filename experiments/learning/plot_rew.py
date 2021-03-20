import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, title="PPO Takeoff Rewards",
                     xlabel="steps", ylabel="rewards")

with np.load(os.path.join('experiments/learning/results/save-takeoff-ppo-kin-dyn-ori_2-03.19.2021_19.02.07', 'evaluations.npz')) as data:
    ax.plot(data["timesteps"], data["results"][:, 0], label="hoge", c="b")
    ax.legend()
plt.show()
