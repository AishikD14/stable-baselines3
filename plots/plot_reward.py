import numpy as np
import matplotlib.pyplot as plt
import os

env = "Ant-v5"
# env = "AntDir-v0"

start_iteration = 5000

# Ant-v5
plot_list = [
    ["PPO_empty_space_optimal_5M_1", "Neighbor Sampling+Empty Space (gamma=0.9)"],
    ["PPO_empty_space_optimal_5M_2", "Neighbor Sampling+Empty Space(gamma=0.5)"],
    ["PPO_neighbor_search_random_walk_optimal_5M_1", "Neighbor Sampling+Random Walk"],
    ["PPO_random_search_random_walk_optimal_5M_1", "Random Sample+Random Walk"],
    ["PPO_normal_train_optimal_5M_1", "Normal Training"]
]

# AntDir-v0
# plot_list = [
#     ["PPO_empty_space_optimal_5M_1", "Neighbor Sampling+Empty Space (gamma=0.9)"],
#     ["PPO_empty_space_optimal_5M_2_1", "Neighbor Sampling+Empty Space(gamma=0.5)"],
#     ["PPO_empty_space_optimal_5M_3_1", "Neighbor Sampling+Empty Space(gamma=0.3)"],
#     ["PPO_neighbor_search_random_walk_optimal_5M_1", "Neighbor Sampling+Random Walk"],
#     ["PPO_random_search_random_walk_optimal_5M_1", "Random Sample+Random Walk"],
#     ["PPO_normal_train_optimal_5M_1", "Normal Training"]
# ]

# Ant-v5 Less evaluations
# plot_list = [
#     ["PPO_empty_space_ls_1", "PPO_empty_space with 100 iterations - 300k"],
#     ["PPO_empty_space_ls_2", "PPO_empty_space with 60 iterations & 3 evaluation - 120k"],
#     ["PPO_empty_space_ls_3", "PPO_empty_space with 60 iterations & 3 evaluation & every other point - 60k"],
#     ["PPO_empty_space_optimal_5M_1", "PPO_empty_space normal - 1M"],
#     ["PPO_normal_train_optimal_5M_1", "Normal Training"]
# ]

plot_metrics = []

# -------------------------------------------------------

for plot_item in plot_list:
    directory = "../logs/"+env+"/"+plot_item[0]
    print("------------------------------------")
    print("Working on "+plot_item[0]+" directory")
    reward_values = []
    try:
        for filename in os.listdir(directory):
            # Check if the filename starts with "results"
            if filename.startswith('results'):
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

plt.xlabel('Samples (x200)')
plt.ylabel('Max Reward')
plt.title('Reward Plot')
plt.grid()
plt.legend()

plt.savefig('../images/rewards.png')