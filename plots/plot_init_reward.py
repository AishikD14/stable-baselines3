import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant_dir, args_ant, args_hopper, args_half_cheetah, args_walker2d, args_humanoid, args_cartpole, args_mountain_car, args_pendulum

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "AntDir"
# env = "CartPole-v1"
# env = "MountainCar-v0"
env = "Pendulum-v1"

if env == "AntDir-v0":
    args = args_ant_dir.get_args(rest_args)
elif env == "Ant-v5":
    args = args_ant.get_args(rest_args)
elif env == "Hopper-v5":
    args = args_hopper.get_args(rest_args)
elif env == "HalfCheetah-v5":
    args = args_half_cheetah.get_args(rest_args)
elif env == "Walker2d-v5":
    args = args_walker2d.get_args(rest_args)
elif env == "Humanoid-v5":
    args = args_humanoid.get_args(rest_args)
elif env == "CartPole-v1":
    args = args_cartpole.get_args(rest_args)
elif env == "MountainCar-v0":
    args = args_mountain_car.get_args(rest_args)
elif env == "Pendulum-v1":
    args = args_pendulum.get_args(rest_args)

# Plot
plt.figure(figsize=(10, 6))

file = "../base_job_output/"+env+"/ppo_init_1M_out.txt"

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

file = "../base_job_output/"+env+"/ppo_init_3M_out.txt"

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
plt.plot(x, smoothed, label="Normal Training (3M)")
plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

# ---------------------------------------------------------------------------------

plt.title("Rewards over Iterations")
plt.xlabel('Samples (x' + str(args.n_steps_per_rollout) + ')')
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.savefig("../images/init_rewards_smooth.png")