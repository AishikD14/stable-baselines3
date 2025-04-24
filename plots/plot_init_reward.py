import numpy as np
import matplotlib.pyplot as plt
import re

# env = "Ant"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "AntDir"

# Plot
plt.figure(figsize=(10, 6))

file = "../base_job_output/"+env+"/ppo_init_1M_1_out.txt"

reward_values = []

with open(file, "r") as f:
    lines = f.readlines()

# Go through each line and extract the reward if it's a reward line
for line in lines:
    if line.startswith("Reward at iter"):
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if match:
            reward = float(match[-1])  # get the last number in the line
            reward_values.append(reward)

# Convert rewards to numpy array for easier math
rewards = np.array(reward_values)

# Smoothing window
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Calculate standard deviation over the same window
stds = np.array([
    np.std(rewards[i:i+window]) if i+window <= len(rewards) else 0
    for i in range(len(smoothed))
])

# X-axis values aligned with smoothed curve
x = np.arange(len(smoothed))

plt.plot(x, smoothed, label="Normal Training (1M)")
plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

# ---------------------------------------------------------------------------------

file = "../base_job_output/"+env+"/ppo_init_5M_out.txt"

reward_values = []

with open(file, "r") as f:
    lines = f.readlines()

# Go through each line and extract the reward if it's a reward line
for line in lines:
    if line.startswith("Reward at iter"):
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if match:
            reward = float(match[-1])  # get the last number in the line
            reward_values.append(reward)

# Convert rewards to numpy array for easier math
rewards = np.array(reward_values)

# Smoothing window
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Calculate standard deviation over the same window
stds = np.array([
    np.std(rewards[i:i+window]) if i+window <= len(rewards) else 0
    for i in range(len(smoothed))
])

# X-axis values aligned with smoothed curve
x = np.arange(len(smoothed))

# Plot
# plt.plot(x, smoothed, label="Normal Training (5M)")
# plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

# ---------------------------------------------------------------------------------

plt.title("Rewards over Iterations")
plt.xlabel("Samples (x512)")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.savefig("../images/init_rewards_smooth.png")