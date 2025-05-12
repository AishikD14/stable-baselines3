import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Walker2d-v5"
env = "Humanoid-v5"
# env = "Swimmer-v5"
# env = "Pendulum-v1"
# env = "BipedalWalker-v3"

if env == "Ant-v5":
    args = args_ant.get_args(rest_args)
elif env == "HalfCheetah-v5":
    args = args_half_cheetah.get_args(rest_args)
elif env == "Walker2d-v5":
    args = args_walker2d.get_args(rest_args)
elif env == "Humanoid-v5":
    args = args_humanoid.get_args(rest_args)
elif env == "Swimmer-v5":
    args = args_swimmer.get_args(rest_args)

# start_iteration = 1
start_iteration = 1000000 // args.n_steps_per_rollout

plot_list = [
    ["PPO_upper_bound", "ExploRLer+PPO"],
    ["PPO_normal_training", "PPO"],
    ["TRPO_normal_training", "TRPO"],
]

plot_metrics = []

# -------------------------------------------------------

for plot_item in plot_list:
    file_path = "../combined_results/"+env+"/"+plot_item[0]+".npy"
    print("------------------------------------")
    print("Working on "+plot_item[0]+" directory")
    
    # Convert rewards to numpy array for easier math
    rewards = np.load(file_path)
    print("Rewards shape: ", rewards.shape)

    # Smoothing window
    window = 100
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    # Calculate standard deviation over the same window
    stds = np.array([
        np.std(rewards[i:i+window]) if i+window <= len(rewards) else 0
        for i in range(len(smoothed))
    ])

    x = np.arange(start_iteration, start_iteration+len(smoothed))

    plot_metrics.append([x, smoothed, stds, len(smoothed)])

# -------------------------------------------------------

# plt.style.use('neurips.mplstyle')
line_styles = ['-', '--', '-.']
plt.figure(figsize=(10, 6))
# Find the minimum length of x
min_length = min([len(x) for x, _, _, _ in plot_metrics])
for i, plot_metric in enumerate(plot_metrics):
    x, smoothed, stds, length = plot_metric
    # Adjust x to the minimum length
    x = x[:min_length]
    smoothed = smoothed[:min_length]
    stds = stds[:min_length]
    
    line_style = line_styles[i % len(line_styles)]
    plt.plot(x, smoothed, label=plot_list[i][1], linestyle=line_style)
    plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

ax = plt.gca()
ax.set_facecolor('#f5f5f5')

plt.xlabel('Samples (x' + str(args.n_steps_per_rollout) + ')')
plt.ylabel('Average Return')
plt.title(env)
plt.grid(True, color='white')

legend = plt.legend()
legend.get_frame().set_facecolor('#f5f5f5')

plt.savefig('../images/paper2.png')