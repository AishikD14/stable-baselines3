import numpy as np
import matplotlib.pyplot as plt
import re

env = "Ant"
# env = "AntDir"

start_iteration = 1

file = "../base_job_output/"+env+"/ppo_empty_space_ls_6_out.txt"

perason_reward_values = []
spearman_reward_values = []

# Plot
plt.figure(figsize=(10, 6))

with open(file, "r") as f:
    lines = f.readlines()

# ------------------------------------------------------------------------------

# Go through each line and extract the reward if it's a reward line
for line in lines:
    if line.startswith("Pearson correlation coefficient"):
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if match:
            reward = float(match[-1])  # get the last number in the line
            perason_reward_values.append(reward)

# Convert rewards to numpy array for easier math
rewards = np.array(perason_reward_values)

# Smoothing window
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Calculate standard deviation over the same window
stds = np.array([
    np.std(rewards[i:i+window]) if i+window <= len(rewards) else 0
    for i in range(len(smoothed))
])

# X-axis values aligned with smoothed curve
x = np.arange(start_iteration, start_iteration+len(smoothed))

plt.plot(x, smoothed, label="Pearson correlation")
plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

# ------------------------------------------------------------------------------

# Go through each line and extract the reward if it's a reward line
for line in lines:
    if line.startswith("Spearman correlation coefficient"):
        match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if match:
            reward = float(match[-1])  # get the last number in the line
            spearman_reward_values.append(reward)

# Convert rewards to numpy array for easier math
rewards = np.array(spearman_reward_values)

# Smoothing window
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Calculate standard deviation over the same window
stds = np.array([
    np.std(rewards[i:i+window]) if i+window <= len(rewards) else 0
    for i in range(len(smoothed))
])

# X-axis values aligned with smoothed curve
x = np.arange(start_iteration, start_iteration+len(smoothed))

plt.plot(x, smoothed, label="Spearman correlation")
plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

# ------------------------------------------------------------------------------

plt.title("Rewards over Iterations")
plt.xlabel("Samples (x512)")
plt.ylabel("Reward")
plt.grid()
plt.legend()

plt.savefig('../images/correlation.png')