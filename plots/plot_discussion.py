import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker
from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
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
    ["PPO_discussion_1"],
]

plot_metrics = []
reward_values = []
adv_reward_values = []

# -------------------------------------------------------

for plot_item in plot_list:
    directory = "../logs/"+env+"/"+plot_item[0]
    print("------------------------------------")
    print("Working on "+plot_item[0]+" directory")

    count = 0

    for filename in os.listdir(directory):
        if count > 10:
            break
        # Check if the filename starts with "results"
        if filename.startswith("results"):
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Convert to list
            results = results.tolist()
            # Append all the reward to the rewards list
            reward_values.extend(results)

            count += 1
    
    # Convert rewards to numpy array for easier math
    rewards = np.array(reward_values)
    print("Rewards shape: ", rewards.shape)

    count = 0

    for filename in os.listdir(directory):
        if count > 10:
            break
        # Check if the filename starts with "results"
        if filename.startswith("adv_results"):
            # Load the rewards
            adv_results = np.load(directory + "/" + filename)
            # Convert to list
            adv_results = adv_results.tolist()
            # Append all the reward to the rewards list
            adv_reward_values.extend(adv_results)

            count += 1
    
    # Convert rewards to numpy array for easier math
    adv_rewards = np.array(adv_reward_values)
    print("Adv Rewards shape: ", adv_rewards.shape)

    fqe_estimates = adv_rewards
    true_rewards = rewards

    # -----------------------------------------------------------------------------------

    # Scatter plot
    plt.figure(figsize=(10,6))
    plt.scatter(adv_rewards, rewards, alpha=0.7)
    plt.xlabel("FQE Estimated Policy Value")
    plt.ylabel("Online Evaluation Return")
    plt.title("FQE Estimates vs True Returns")

    # Fit a linear regression line (1st degree polynomial)
    # slope, intercept = np.polyfit(rewards, adv_rewards, 1)
    # line_x = np.linspace(min(rewards), max(rewards), 100)
    # line_y = slope * line_x + intercept
    # plt.plot(line_x, line_y, 'r--')

    # Correlation metrics
    pearson_r, _ = pearsonr(rewards, adv_rewards)
    spearman_r, _ = spearmanr(rewards, adv_rewards)
    plt.legend(title=f"Spearman ρ = {spearman_r:.2f}", loc='lower right')
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('../paper_plots/discussion.png')

    # -------------------------------------------------------------------------------------

    # num_per_iter = 20
    # iterations = np.arange(len(true_rewards)) // num_per_iter

    # # Compute correlation
    # pearson_r, _ = pearsonr(fqe_estimates, true_rewards)
    # spearman_r, _ = spearmanr(fqe_estimates, true_rewards)

    # # Plot
    # plt.figure(figsize=(10,6))
    # scatter = plt.scatter(fqe_estimates, true_rewards, c=iterations, cmap='viridis', alpha=0.8)
    # cbar = plt.colorbar(scatter)
    # cbar.set_label("Training Iteration")

    # # Optional: Add regression line
    # slope, intercept = np.polyfit(fqe_estimates, true_rewards, 1)
    # x_vals = np.linspace(min(fqe_estimates), max(fqe_estimates), 100)
    # y_vals = slope * x_vals + intercept
    # plt.plot(x_vals, y_vals, 'r--')

    # plt.xlabel("FQE Estimated Return")
    # plt.ylabel("True Online Reward")
    # plt.title("FQE vs True Reward, Colored by Iteration")
    # plt.legend(title=f"Pearson r = {pearson_r:.2f}\nSpearman ρ = {spearman_r:.2f}")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('../images/discussion.png')

    # --------------------------------------------------------------------------------------------------

    # policy_ids = np.arange(len(true_rewards))  # 0 to 139

    # plt.figure(figsize=(10,5))

    # plt.plot(policy_ids, true_rewards, label="True Reward", linestyle='-')
    # plt.plot(policy_ids, fqe_estimates, label="FQE Estimate", linestyle='--')

    # plt.xlabel("Policy Number (Checkpoint Index)")
    # plt.ylabel("Reward")
    # plt.title("True vs FQE Estimated Rewards Over Policies")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('../images/discussion.png')

    # ---------------------------------------------------------------------------------------------------