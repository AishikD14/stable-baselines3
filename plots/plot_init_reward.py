import numpy as np
import matplotlib.pyplot as plt
import os
import re

file = "../base_job_output/ppo_init_1M_out.txt"

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

# print("All rewards:", reward_values)

# Plot the rewards
plt.plot(reward_values, label="Normal Training")
plt.title("Rewards over iterations")
plt.xlabel("Iteration (x10000)")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.savefig('../images/init_rewards.png')