import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker, args_lunarlander, args_hopper, args_fetch_reach, args_fetch_reach_dense, args_fetch_push, args_fetch_push_dense
import re

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "Swimmer-v5"
# env = "Pendulum-v1"
# env = "BipedalWalker-v3"
# env = "LunarLander-v3"
# env = "FetchReach-v4"
# env = "FetchReachDense-v4"
# env = "FetchPush-v4"
# env = "FetchPushDense-v4"

if env == "Ant-v5":
    args = args_ant.get_args(rest_args)
elif env == "HalfCheetah-v5":
    args = args_half_cheetah.get_args(rest_args)
elif env == "Hopper-v5":
    args = args_hopper.get_args(rest_args)
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
elif env == "FetchReach-v4":
    args = args_fetch_reach.get_args(rest_args)
elif env == "FetchReachDense-v4":
    args = args_fetch_reach_dense.get_args(rest_args)
elif env == "FetchPush-v4":
    args = args_fetch_push.get_args(rest_args)
elif env == "FetchPushDense-v4":
    args = args_fetch_push_dense.get_args(rest_args)

# start_iteration = 1000000 // args.n_steps_per_rollout*1
start_iteration = 1

if env in ["Ant-v5", "Walker2d-v5", "Humanoid-v5"]:
    plot_list = [
        ["PPO_upper_bound", "ExploRLer-P"],
        # ["PPO_upper_bound_interpolated", "ExploRLer-P"],
        # ["PPO_NoPretrain", "ExploRLer-P No Pre-training"],
        # ["PPO_NoPretrain_interpolated", "ExploRLer-P No Pre-training"],
        ["PPO_normal_training", "PPO"],
        # ["PPO_normal_training_interpolated", "PPO"],
        # ["SAC_normal_training", "SAC"],
        # ["TRPO_upper_bound", "ExploRLer-T"],
        ["TRPO_upper_bound_interpolated", "ExploRLer-T"],
        # ["TRPO_normal_training", "TRPO"],
        ["TRPO_normal_training_interpolated", "TRPO"],
        # ["PPO_empty_space_ls", "FQE Estimation"],
        # ["PPO_baseline", "Baseline"],
        # ["PPO_upper_bound", "Online Evaluation"],
        ["PPO_PBT", "PBT"],
        # ["PPO_PBT_interpolated", "PBT"],
        # ["TRPO_PBT", "PBT"],
        # ["TRPO_PBT_interpolated", "PBT"],
        ["PPO_VFS", "VFS"],
        # ["PPO_VFS_interpolated", "VFS"],
        # ["TRPO_VFS", "VFS"],
        # ["TRPO_VFS_interpolated", "VFS"],
        ["PPO_GuidedES", "Guided ES"],
        # ["PPO_GuidedES_interpolated", "Guided ES"],
        # ["TRPO_GuidedES", "Guided ES"],
        # ["TRPO_GuidedES_interpolated", "Guided ES"],
        # ["PPO_neghrand", "Random Walk"],
        # ["PPO_neghrand_interpolated", "Random Walk"],
        ["PPO_CheckpointAvg", "Checkpoint Avg"],
        # ["PPO_CheckpointAvg_interpolated", "Checkpoint Avg"],
        # ["TRPO_CheckpointAvg", "Checkpoint Avg"],
        # ["TRPO_CheckpointAvg_interpolated", "Checkpoint Avg"],
    ]

elif env == "Hopper-v5":
    plot_list = [
        # ["PPO_upper_bound", "ExploRLer-P"],
        ["PPO_upper_bound_interpolated", "ExploRLer-P"],
        # ["PPO_NoPretrain", "ExploRLer-P No Pre-training"],
        # ["PPO_NoPretrain_interpolated", "ExploRLer-P No Pre-training"],
        ["PPO_normal_training", "PPO"],
        # ["PPO_normal_training_interpolated", "PPO"],
        # ["SAC_normal_training", "SAC"],
        # ["TRPO_upper_bound", "ExploRLer-T"],
        ["TRPO_upper_bound_interpolated", "ExploRLer-T"],
        # ["TRPO_normal_training", "TRPO"],
        ["TRPO_normal_training_interpolated", "TRPO"],
        # ["PPO_empty_space_ls", "FQE Estimation"],
        # ["PPO_baseline", "Baseline"],
        # ["PPO_upper_bound", "Online Evaluation"],
        # ["PPO_PBT", "PBT"],
        ["PPO_PBT_interpolated", "PBT"],
        # ["TRPO_PBT", "PBT"],
        # ["TRPO_PBT_interpolated", "PBT"],
        # ["PPO_VFS", "VFS"],
        ["PPO_VFS_interpolated", "VFS"],
        # ["TRPO_VFS", "VFS"],
        # ["TRPO_VFS_interpolated", "VFS"],
        # ["PPO_GuidedES", "Guided ES"],
        ["PPO_GuidedES_interpolated", "Guided ES"],
        # ["TRPO_GuidedES", "Guided ES"],
        # ["TRPO_GuidedES_interpolated", "Guided ES"],
        # ["PPO_neghrand", "Random Walk"],
        # ["PPO_neghrand_interpolated", "Random Walk"],
        # ["PPO_CheckpointAvg", "Checkpoint Avg"],
        ["PPO_CheckpointAvg_interpolated", "Checkpoint Avg"],
        # ["TRPO_CheckpointAvg", "Checkpoint Avg"],
        # ["TRPO_CheckpointAvg_interpolated", "Checkpoint Avg"],
    ]

elif env == "HalfCheetah-v5":
    plot_list = [
        ["PPO_upper_bound", "ExploRLer-P"],
        # ["PPO_upper_bound_interpolated", "ExploRLer-P"],
        # ["PPO_NoPretrain", "ExploRLer-P No Pre-training"],
        # ["PPO_NoPretrain_interpolated", "ExploRLer-P No Pre-training"],
        ["PPO_normal_training", "PPO"],
        # ["PPO_normal_training_interpolated", "PPO"],
        # ["SAC_normal_training", "SAC"],
        # ["TRPO_upper_bound", "ExploRLer-T"],
        ["TRPO_upper_bound_interpolated", "ExploRLer-T"],
        # ["TRPO_normal_training", "TRPO"],
        ["TRPO_normal_training_interpolated", "TRPO"],
        # ["PPO_empty_space_ls", "FQE Estimation"],
        # ["PPO_baseline", "Baseline"],
        # ["PPO_upper_bound", "Online Evaluation"],
        # ["PPO_PBT", "PBT"],
        # ["PPO_PBT_interpolated", "PBT"],
        # ["TRPO_PBT", "PBT"],
        ["TRPO_PBT_interpolated", "PBT"],
        # ["PPO_VFS", "VFS"],
        # ["PPO_VFS_interpolated", "VFS"],
        # ["TRPO_VFS", "VFS"],
        ["TRPO_VFS_interpolated", "VFS"],
        # ["PPO_GuidedES", "Guided ES"],
        # ["PPO_GuidedES_interpolated", "Guided ES"],
        # ["TRPO_GuidedES", "Guided ES"],
        ["TRPO_GuidedES_interpolated", "Guided ES"],
        # ["PPO_neghrand", "Random Walk"],
        # ["PPO_neghrand_interpolated", "Random Walk"],
        # ["PPO_CheckpointAvg", "Checkpoint Avg"],
        # ["PPO_CheckpointAvg_interpolated", "Checkpoint Avg"],
        # ["TRPO_CheckpointAvg", "Checkpoint Avg"],
        ["TRPO_CheckpointAvg_interpolated", "Checkpoint Avg"],
    ]

elif env in ["Pendulum-v1", "BipedalWalker-v3"]:
    plot_list = [
        ["PPO_upper_bound", "ExploRLer-P"],
        # ["PPO_upper_bound_interpolated", "ExploRLer-P"],
        ["PPO_normal_training", "PPO"],
        # ["PPO_normal_training_interpolated", "PPO"],
        # ["TRPO_upper_bound", "ExploRLer-T"],
        ["TRPO_upper_bound_interpolated", "ExploRLer-T"],
        ["TRPO_normal_training", "TRPO"],
        # ["TRPO_normal_training_interpolated", "TRPO"],
    ]
# ==================================================================================================

#  For SAC plots
# plot_list = [
#     ["SAC_upper_bound", "ExploRLer-S"],
#     ["SAC_normal_training", "SAC"]
# ]

# ==================================================================================================

# For Ablation study plots (No Pre-training)
# if env in ["Ant-v5", "Walker2d-v5", "Humanoid-v5", "Pendulum-v1", "BipedalWalker-v3"]:
#     plot_list = [
#         ["PPO_upper_bound", "ExploRLer with Pre-training"],
#         # ["PPO_upper_bound_interpolated", "ExploRLer-P"],
#         ["PPO_NoPretrain", "ExploRLer without Pre-training"],
#         # ["PPO_NoPretrain_interpolated", "ExploRLer-P No Pre-training"],
#         # ["PPO_normal_training", "PPO"],
#         # ["PPO_normal_training_interpolated", "PPO"],
#     ]

# elif env == "Hopper-v5":
#     plot_list = [
#         # ["PPO_upper_bound", "ExploRLer-P"],
#         ["PPO_upper_bound_interpolated", "ExploRLer-P"],
#         # ["PPO_NoPretrain", "ExploRLer-P No Pre-training"],
#         ["PPO_NoPretrain_interpolated", "ExploRLer-P No Pre-training"],
#         # ["PPO_normal_training", "PPO"],
#         # ["PPO_normal_training_interpolated", "PPO"],
#     ]

# elif env == "HalfCheetah-v5":
#     plot_list = [
#         # ["PPO_upper_bound", "ExploRLer-P"],
#         # ["PPO_upper_bound_interpolated", "ExploRLer-P"],
#         # ["PPO_NoPretrain", "ExploRLer-P No Pre-training"],
#         # ["PPO_NoPretrain_interpolated", "ExploRLer-P No Pre-training"],
#         # ["PPO_normal_training", "PPO"],
#         # ["PPO_normal_training_interpolated", "PPO"],
#         # ["SAC_normal_training", "SAC"],
#         # ["TRPO_upper_bound", "ExploRLer-T"],
#         ["TRPO_upper_bound_interpolated", "ExploRLer"],
#         ["TRPO_NoPretrain_interpolated", "ExploRLer No Pre-training"],
#         # ["TRPO_normal_training", "TRPO"],
#         # ["TRPO_normal_training_interpolated", "TRPO"],
        
    # ]

# ==================================================================================================

# For Ablation study plots (m=[3,4,5], E=[3,6,10], I=[10,20,30,40])
if env in ["Ant-v5", "Walker2d-v5", "Humanoid-v5", "Pendulum-v1", "BipedalWalker-v3"]:
    plot_list = [
        # ["PPO_upper_bound", "I=10"],
        # ["PPO_hyper_I_20_interpolated", "I=20"],
        # ["PPO_hyper_I_30_interpolated", "I=30"],
        # ["PPO_hyper_I_40_interpolated", "I=40"],
        # ["PPO_upper_bound", "m=5"],
        # ["PPO_hyper_m_4", "m=4"],
        # ["PPO_hyper_m_3", "m=3"],
        # ["PPO_upper_bound", "E=3"],
        # ["PPO_hyper_E_6", "E=6"],
        # ["PPO_hyper_E_10", "E=10"],
        ["PPO_upper_bound", "m=5,E=3"],
        ["PPO_hyper_m_3_E_6", "m=3,E=6"],
        # ["PPO_upper_bound", "PPO Normal Training"],
        # ["PPO_parallel", "PPO Parallel Training"],
    ]

# ==================================================================================================

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

    if  env in ["Pendulum-v1", "BipedalWalker-v3", "Swimmer-v5", "FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
        window = 10

    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    # Calculate standard deviation over the same window
    stds = np.array([
        np.std(rewards[i:i+window]) if i+window <= len(rewards) else 0
        for i in range(len(smoothed))
    ])

    x = np.arange(start_iteration, start_iteration+len(smoothed))

    plot_metrics.append([x, smoothed, stds, len(smoothed)])

# -------------------------------------------------------

# line_styles = ['-', '--', '-', '--', '--', '--', '--', '--'] # Normal
line_styles = ['-', '--', '--', '--'] # Ablation

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
    
    plt.plot(x, smoothed, label=plot_list[i][1], linestyle=line_styles[i])
    plt.fill_between(x, smoothed - stds, smoothed + stds, alpha=0.2)

ax = plt.gca()
ax.set_facecolor('#f5f5f5')

plt.xlabel('Iterations', fontsize=25)
if env in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
    plt.ylabel('Average Success Rate', fontsize=20)
else:
    plt.ylabel('Average Return', fontsize=25)
    if env not in  ["Ant-v5", "BipedalWalker-v3", "Pendulum-v1"]:
        ax.yaxis.label.set_visible(False)

plt.title(env, fontsize=30)
plt.grid(True, color='white')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# --------------------------------------------------------------------------------------------------

legend = plt.legend(fontsize=16)
# legend.get_frame().set_facecolor('#f5f5f5')

# handles, labels = ax.get_legend_handles_labels()

# # Create a new figure just for the legend
# fig_legend = plt.figure(figsize=(6, 6))  # adjust size as needed
# leg = fig_legend.legend(handles, labels, loc='center', ncol=1, frameon=False, fontsize=20)
# leg.get_frame().set_facecolor('#f5f5f5')

# # Save only the legend
# fig_legend.savefig("../paper_plots/legend.png", bbox_inches='tight', dpi=300)
# plt.close(fig_legend)

# quit()

# --------------------------------------------------------------------------------------------------

for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig('../paper_plots/'+env+'_m[3,5]_E[6,3].pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('../paper_plots/'+env+'_m[3,5]_E[6,3].png', bbox_inches='tight', dpi=300)