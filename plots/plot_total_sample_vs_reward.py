import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant_dir, args_ant, args_hopper, args_half_cheetah, args_walker2d, args_humanoid, args_cartpole, args_mountain_car, args_pendulum

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "AntDir-v0"
# env = "CartPole-v1"
# env = "MountainCar-v0"
# env = "Pendulum-v1"

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

# start_iteration = 1
start_iteration = 1000000 // args.n_steps_per_rollout

# Ant-v5
# plot_list = [
    # ["PPO_empty_space_optimal_5M_1", "Neighbor Sampling+Empty Space (gamma=0.9)"],
#     ["PPO_empty_space_optimal_5M_2", "Neighbor Sampling+Empty Space(gamma=0.5)"],
#     ["PPO_neighbor_search_random_walk_optimal_5M", "Neighbor Sampling+Random Walk"],
#     ["PPO_random_search_random_walk_optimal_5M", "Random Sample+Random Walk"]
# ]

# AntDir-v0
# plot_list = [
#     ["PPO_empty_space_optimal_5M_1", "Neighbor Sampling+Empty Space (gamma=0.9)"],
#     ["PPO_empty_space_optimal_5M_2_1", "Neighbor Sampling+Empty Space(gamma=0.5)"],
#     ["PPO_empty_space_optimal_5M_3_1", "Neighbor Sampling+Empty Space(gamma=0.3)"],
#     ["PPO_neighbor_search_random_walk_optimal_5M_1", "Neighbor Sampling+Random Walk"],
#     ["PPO_random_search_random_walk_optimal_5M_1", "Random Sample+Random Walk"],
#     ["PPO_normal_train_optimal_5M", "Normal Training"]
# ]

# Ant-v5 Less evaluations
# plot_list = [
    # ["PPO_empty_space_ls_1", "PPO_empty_space with 100 iterations-300k(gamma=0.9)"],
    # ["PPO_empty_space_ls_2", "PPO_empty_space with 60 iterations & 3 evaluation-120k(gamma=0.9)"],
    # ["PPO_empty_space_ls_3", "PPO_empty_space with 60 iterations & 3 evaluation & every other point-60k(gamma=0.9)"],
    # ["PPO_empty_space_optimal_5M_1", "PPO_empty_space normal-1M(gamma=0.9)"],
    # ["PPO_empty_space_ls_7", "PPO_empty_space with 60 iterations & 3 evaluation & every other point-60k(gamma=0.5)"],
    # ["PPO_empty_space_optimal_5M_2", "PPO_empty_space normal-1M(gamma=0.5)"],
    # ["PPO_empty_space_ls_8", "PPO_empty_space with 60 iterations & 3 evaluation & every other point-60k(gamma=0.3)"],
# ]

# Ant-v5 FQE evaluations
plot_list = [
    # ["PPO_empty_space_ls_11", "FQE top 5 agent & 2 eval & search=1 -10k; gamma=0.3"],
    # ["PPO_baseline_1", "Baseline with search=1 -10k"],
    # ["PPO_empty_space_ls_12", "FQE top 5 agent & 2 eval & search=2- 5k; gamma=0.3"],
    # ["PPO_baseline_2", "Baseline with search=2 -5k"],
    # ["PPO_empty_space_ls_13", "FQE top 5 agent & 2 eval & search=3- 3k; gamma=0.3"],
    # ["PPO_baseline_3", "Baseline with search=3 -3k"],
    ["PPO_FQE_1", "PPO FQE with 60 iterations & every other point; gamma=0.3"]
    # ["PPO_empty_space_ls_8", "Online eval with 60 iter & 3 eval & every other point -60k; gamma=0.3"], # Upper bound
]

# HalfCheetah-v5 FQE evaluations
# plot_list = [
#     ["PPO_empty_space_ls_1", "FQE top 5 agent and 2 eval & search=1- 10k; gamma=0.3"],
#     ["PPO_baseline_1", "Baseline with search=1 -10k"],
#     # ["PPO_empty_space_ls_2", "FQE top 5 agent and 2 eval & search=2- 5k; gamma=0.3"],
#     # ["PPO_baseline_2", "Baseline with search=2 -5k"],
#     # ["PPO_empty_space_ls_3", "FQE top 5 agent and 2 eval & search=3- 3k; gamma=0.3"],
#     # ["PPO_baseline_3", "Baseline with search=3 -3k"],
#     ["PPO_empty_space", "Online eval with 60 iter & 3 eval & every other point-60k; gamma=0.3"], # Upper bound
# ]

# Hopper-v5 FQE evaluations
# plot_list = [
#     ["PPO_empty_space_ls_1", "FQE top 5 agent and 2 eval & search=2- 5k; gamma=0.3"],
#     ["PPO_baseline_1", "Baseline with search=2 -5k"],
#     ["PPO_empty_space_1", "Online eval with 60 iter & 3 eval & every other point-60k; gamma=0.3"], # Upper bound
# ]

# Walker2d-v5 FQE evaluations
# plot_list = [
#     ["PPO_empty_space_1", "Online eval with 60 iter & 3 eval & every other point-60k; gamma=0.3"], # Upper bound
# ]

# Humanoid-v5 FQE evaluations
# plot_list = [
#     # ["PPO_empty_space_ls_1", "FQE top 5 agent and 2 eval & search=1- 10k; gamma=0.3"],
#     # ["PPO_baseline_1", "Baseline with search=1 -10k"],
#     ["PPO_empty_space_ls_2", "FQE top 5 agent and 2 eval & search=2- 5k; gamma=0.3"],
#     ["PPO_baseline_2", "Baseline with search=2 -5k"],
#     # ["PPO_empty_space_ls_3", "FQE top 5 agent and 2 eval & search=3- 3k; gamma=0.3"],
#     # ["PPO_baseline_3", "Baseline with search=3 -3k"],
#     ["PPO_empty_space_1", "Online eval with 60 iter & 3 eval & every other point-60k; gamma=0.3"], # Upper bound
# ]

# CartPole-v1 FQE evaluations
# plot_list = [
    # ["PPO_empty_space_ls_1", "FQE top 5 agent and 2 eval & search=2- 10k; gamma=0.3"],
    # ["PPO_baseline_1", "Baseline with search=2 -10k"],
    # ["PPO_empty_space_1", "Online eval with 60 iter & 3 eval & every other point-60k; gamma=0.3"], # Upper bound
# ]

# Pendulum-v1 FQE evaluations
# plot_list = [
#     # ["PPO_empty_space_ls_1", "FQE top 5 agent and 2 eval & search=2- 10k; gamma=0.3"],
#     ["PPO_baseline_1", "Baseline with search=1 -10k"],
#     # ["PPO_empty_space_1", "Online eval with 60 iter & 3 eval & every other point-60k; gamma=0.3"], # Upper bound
# ]

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
plt.ylabel('Max Reward')
plt.title('Reward Plot')
plt.grid()
plt.legend()

plt.savefig('../images/rewards.png')