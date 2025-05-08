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
env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "Swimmer-v5"

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

seed_list = [0, 1, 2]
# file_name = "PPO_normal_training"

plot_list = [
    # ["PPO_FQE", "PPO FQE with 60 iterations & every other point; gamma=0.3"],
    ["PPO_normal_training", "PPO Normal Training"],
    # ["TRPO_normal_training", "TRPO Normal Training"],
]

plot_metrics = []

for plot_item in plot_list:

    all_rewards = []

    for i in seed_list:
        directory = "../final_results/"+env+"/"+plot_item[0]+"_"+str(i+1)
        print("------------------------------------")
        print("Working on "+plot_item[0]+"_"+str(i+1)+" directory")
        
        results = np.load(directory + ".npy")

        # Append the rewards to all_rewards
        all_rewards.append(results)

    # Convert all_rewards to numpy array for easier math
    all_rewards_np = np.stack(all_rewards)
    print(all_rewards_np.shape)

    # Calculate the mean and standard deviation across seeds
    mean_rewards = np.mean(all_rewards_np, axis=0)
    print(mean_rewards.shape)

    # Save the mean rewards
    os.makedirs("../combined_results/"+env, exist_ok=True)
    np.save("../combined_results/"+env+"/"+plot_item[0]+".npy", mean_rewards)

    # Smoothing window
    window = 10
    smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')

    # Calculate standard deviation over the same window
    stds = np.array([
        np.std(mean_rewards[i:i+window]) if i+window <= len(mean_rewards) else 0
        for i in range(len(smoothed))
    ])

    x = np.arange(start_iteration, start_iteration+len(smoothed))

    plot_metrics.append([x, smoothed, stds, len(smoothed)])

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