import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, title="Hover Rewards",
                     xlabel="steps", ylabel="rewards")

with np.load(os.path.join('experiments/learning/results/save-hover-ppo-kin-dyn-ori_2-03.20.2021_14.09.05/', 'evaluations.npz')) as data:
    ax.plot(data["timesteps"], data["results"]
            [:, 0], label="ppo", c="blue")

with np.load(os.path.join('experiments/learning/results/save-hover-a2c-kin-dyn-ori_2-03.20.2021_15.42.05', 'evaluations.npz')) as data:
    ax.plot(data["timesteps"], data["results"][:, 0], label="a2c", c="orange")

ax.legend()
ax.grid(axis='both')
plt.show()
