import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker, args_lunarlander

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
# env = "Walker2d-v5"
env = "Humanoid-v5"
# env = "Swimmer-v5"
# env = "Pendulum-v1"
# env = "BipedalWalker-v3"
# env = "LunarLander-v3"

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
elif env == "Pendulum-v1":
    args = args_pendulum.get_args(rest_args)
elif env == "BipedalWalker-v3":
    args = args_bipedal_walker.get_args(rest_args)
elif env == "LunarLander-v3":
    args = args_lunarlander.get_args(rest_args)

if not hasattr(args, 'n_envs'):
    args.n_envs = 1

# if env in ["Pendulum-v1", "BipedalWalker-v3"]:
#     start_iteration = 1 // args.n_steps_per_rollout*args.n_envs
# else:
#     start_iteration = 1000000 // args.n_steps_per_rollout*args.n_envs

start_iteration = 1

# seed_list = [0]
# seed_list = [0, 1, 2]
seed_list = [0, 1, 2, 3]
# file_name = "PPO_normal_training"

plot_list = [
    # ["PPO_FQE", "PPO FQE with 60 iterations & every other point; gamma=0.3"],
    # ["PPO_normal_training", "PPO Normal Training"],
    # ["PPO_upper_bound", "PPO Upper Bound"],
    # ["TRPO_normal_training", "TRPO Normal Training"],
    # ["TRPO_upper_bound", "TRPO Upper Bound"],
    # ["PPO_Ablation1", "PPO_Ablation1"],
    # ["PPO_Ablation2", "PPO Ablation 2"],
    # ["PPO_Ablation3", "PPO_Ablation3"]
    # ["PPO_Ablation4", "PPO Ablation 4"],
    # ["PPO_Ablation5", "PPO Ablation 5"],
    # ["TRPO_Ablation1", "TRPO_Ablation1"],
    # ["TRPO_Ablation2", "TRPO_Ablation2"],
    # ["TRPO_Ablation5", "TRPO_Ablation5"],
    # ["PPO_neghrand", "PPO Random Walk"],
    # ["TRPO_neghrand", "TRPO Random Walk"],
    # ["PPO_FQE", "PPO FQE"],
    # ["PPO_empty_space_ls", "PPO FQE"],
    # ["PPO_baseline", "PPO Baseline"],
    # ["PPO_CheckpointAvg", "PPO Checkpoint averaging"],
    # ["PPO_PBT", "Population Based Training"],
    ["PPO_NoPretrain", "PPO No-Pretraining"],
    # ["PPO_Rebuttal_4", "Guided ES"],
    # ["PPO_Rebuttal_5", "VFS"],
]

plot_metrics = []

for plot_item in plot_list:

    all_rewards = []

    for i in seed_list:
        directory = "../final_results/"+env+"/"+plot_item[0]+"_"+str(i+1)
        print("------------------------------------")
        print("Working on "+plot_item[0]+"_"+str(i+1)+" directory")
        
        results = np.load(directory + ".npy")
        print(results.shape)

        if env not in ["Pendulum-v1", "BipedalWalker-v3", "LunarLander-v3"] and plot_item[0] not in ["PPO_NoPretrain", "TRPO_NoPretrain"]:
            # Load pretrained rewards
            if "PPO" in plot_item[0]:
                pretrain_rewards = np.load("../final_results/"+env+"/PPO_pretrain_"+str(i+1)+".npy")
            elif "TRPO" in plot_item[0]:
                pretrain_rewards = np.load("../final_results/"+env+"/TRPO_pretrain_"+str(i+1)+".npy")

            # Add pretrain rewards to the beginning of results
            results = np.concatenate((pretrain_rewards, results))
            print(results.shape)

        # results = pretrain_rewards

        # Append the rewards to all_rewards
        all_rewards.append(results)

    # Convert all_rewards to numpy array for easier math
    all_rewards_np = np.stack(all_rewards)
    print(all_rewards_np.shape)

    # Calculate the mean and standard deviation across seeds
    mean_rewards = np.mean(all_rewards_np, axis=0)
    print(mean_rewards.shape)
    std_rewards = np.std(all_rewards_np, axis=0)

    max_idx = np.argmax(mean_rewards)

    # Step 3: Get the corresponding mean and std
    max_avg_reward = mean_rewards[max_idx]
    std_at_max = std_rewards[max_idx]

    # Output the result
    print(f"Max average reward: {max_avg_reward:.2f} ± {std_at_max:.2f} at timestep {max_idx}")

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

# plt.xlabel('Samples (x' + str(args.n_steps_per_rollout*args.n_envs) + ')')
plt.xlabel("Number of Iterations")
plt.ylabel('Average Return')
plt.title('Reward Plot')
plt.grid()
plt.legend()

plt.savefig('../images/paper1.png')