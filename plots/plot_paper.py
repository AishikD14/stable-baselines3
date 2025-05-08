import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
env = "Swimmer-v5"

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
    ["PPO_FQE", "PPO FQE with 60 iterations & every other point; gamma=0.3"],
    ["PPO_normal_training", "PPO Normal Training"],
    ["TRPO_normal_training", "TRPO Normal Training"],
]

plot_metrics = []

# -------------------------------------------------------

for plot_item in plot_list:
    directory = "../logs/"+env+"/"+plot_item[0]
    print("------------------------------------")
    print("Working on "+plot_item[0]+" directory")
    reward_values = []
    searchString = plot_item[2] if len(plot_item) > 2 else "results"
    try:
        for filename in os.listdir(directory):
            # Check if the filename starts with "results"
            if filename.startswith(searchString):
                # Load the rewards
                results = np.load(directory + "/" + filename)
                # Find the maximum reward
                max_reward = np.max(results)
                # Append the maximum reward to the rewards list
                reward_values.append(max_reward)

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

        x = np.arange(start_iteration, start_iteration+len(smoothed))

        plot_metrics.append([x, smoothed, stds, len(smoothed)])

    except :
            print("Error in "+plot_item[0]+" directory")

# -------------------------------------------------------

plt.figure(figsize=(10, 6))
# Find the minimum length of x
min_length = min([len(x) for x, _, _, _ in plot_metrics])
for i, plot_metric in enumerate(plot_metrics):
    x, smoothed, stds, length = plot_metric
    # Adjust x to the minimum length
    x = x[:min_length]
    smoothed = smoothed[:min_length]
    stds = stds[:min_length]
    
    plt.plot(x, smoothed, label=plot_list[i][1])
    plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

plt.xlabel('Samples (x' + str(args.n_steps_per_rollout) + ')')
plt.ylabel('Episode Return')
plt.title('Reward Plot')
plt.grid()
plt.legend()

plt.savefig('../images/paper1.png')