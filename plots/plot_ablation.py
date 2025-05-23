import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker
import re

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
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
elif env == "Pendulum-v1":
    args = args_pendulum.get_args(rest_args)
elif env == "BipedalWalker-v3":
    args = args_bipedal_walker.get_args(rest_args)

start_iteration = 1

# ablation_type = "Ablation1" #gamma
# ablation_type = "Ablation2" #search
# ablation_type = "Ablation3" #neghrand
# ablation_type = "Ablation4" #fqe humanoid
# ablation_type = "Ablation5" #fqe swimmer
ablation_type = "Ablation6" #n_eval

if ablation_type == "Ablation1":
    plot_list = [
        ["PPO_Ablation1", "gamma=0"],
        ["PPO_upper_bound", "gamma=0.3"],
        ["PPO_Ablation2", "gamma=0.5"],
        # ["PPO_Ablation5", "gamma=0.9"],
        # ["TRPO_Ablation1", "gamma=0"],
        # ["TRPO_upper_bound", "gamma=0.3"],
        # ["TRPO_Ablation2", "gamma=0.5"],
        # ["TRPO_Ablation5", "gamma=0.9"],
    ]

if ablation_type == "Ablation2":
    plot_list = [
        ["PPO_upper_bound", "search=1"],
        ["PPO_Ablation3_interpolated", "search=2"],
        ["PPO_Ablation4_interpolated", "search=3"],
    ]

if ablation_type == "Ablation3":
    plot_list = [
        # ["PPO_upper_bound", "Empty-Space Search"],
        # ["PPO_neghrand", "Random Walk"],
        ["TRPO_upper_bound", "Empty-Space Search"],
        ["TRPO_neghrand", "Random Walk"],
    ]

if ablation_type in ["Ablation4", "Ablation5"]:
    plot_list = [
        ["PPO_upper_bound", "Online Evaluation"],
        ["PPO_FQE", "FQE Evaluation"],
    ]

if ablation_type == "Ablation6":
    plot_list = [
        ["PPO_upper_bound", "n_eval=3"],
        ["PPO_eval_1", "n_eval=10"],
        ["PPO_eval_2", "n_eval=20"],
        ["PPO_eval_3", "n_eval=50"],
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
    if env in ["Pendulum-v1", "BipedalWalker-v3", "Swimmer-v5"]:
        window = 10
    else:
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

# plt.xlabel('Samples (x' + str(args.n_steps_per_rollout) + ')')
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Average Return', fontsize=20)

if ablation_type == "Ablation1":
    plt.title(env, fontsize=20)
elif ablation_type == "Ablation2":
    plt.title(env + " with gamma = 0.3", fontsize=20)
else:
    plt.title(env, fontsize=20)

plt.grid(True, color='white')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

legend = plt.legend(fontsize=16)
legend.get_frame().set_facecolor('#f5f5f5')

for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig('../paper_plots/'+env+'_'+ablation_type+'.pdf', format='pdf', bbox_inches='tight', dpi=300)