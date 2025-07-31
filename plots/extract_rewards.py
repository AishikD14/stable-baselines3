import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker, args_lunarlander
import re

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
# env = "HalfCheetah-v5"
env = "Walker2d-v5"
# env = "Humanoid-v5"
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

# Pendulum-v1
file_name_list = [
    # ["PPO_plot_1", "ppo_plot_out", "PPO_pretrain_1"],
    # ["PPO_plot_2", "ppo_plot_1_out", "PPO_pretrain_2"],
    # ["PPO_plot_3", "ppo_plot_2_out", "PPO_pretrain_3"],
    # ["PPO_plot_4", "ppo_plot_3_out", "PPO_pretrain_4"],
    # ["TRPO_plot_1", "trpo_plot_out", "TRPO_pretrain_1"],
    # ["TRPO_plot_2", "trpo_plot_1_out", "TRPO_pretrain_2"],
    # ["TRPO_plot_3", "trpo_plot_2_out", "TRPO_pretrain_3"],
    # ["TRPO_plot_4", "trpo_plot_3_out", "TRPO_pretrain_4"],
    # ["PPO_normal_training_1"],
    # ["PPO_normal_training_2"],
    # ["PPO_normal_training_3"],
    # ["PPO_normal_training_4"],
    # ["PPO_normal_training_7", "PPO_normal_training_4"],
    # ["PPO_upper_bound_1"],
    # ["PPO_upper_bound_2"],
    # ["PPO_upper_bound_3"],
    # ["PPO_upper_bound_4"],
    # ["PPO_upper_bound_7", "PPO_upper_bound_4"],
    # ["TRPO_normal_training_1"],
    # ["TRPO_normal_training_2"],
    # ["TRPO_normal_training_3"],
    # ["TRPO_normal_training_4"],
    # ["TRPO_upper_bound_1"],
    # ["TRPO_upper_bound_2"],
    # ["TRPO_upper_bound_3"],
    # ["TRPO_upper_bound_4"],
    # ["PPO_Ablation1_1"],
    # ["PPO_Ablation1_2"],
    # ["PPO_Ablation1_3"],
    # ["PPO_Ablation2_1"],
    # ["PPO_Ablation2_2"],
    # ["PPO_Ablation2_3"]
    # ["PPO_Ablation3_1", "PPO_Ablation5_1"],
    # ["PPO_Ablation3_2", "PPO_Ablation5_2"],
    # ["PPO_Ablation3_3", "PPO_Ablation5_3"],
    # ["PPO_Ablation4_1"],
    # ["PPO_Ablation4_2"],
    # ["PPO_Ablation4_3"],
    # ["PPO_Ablation5_1"],
    # ["PPO_Ablation5_2"],
    # ["PPO_Ablation5_3"],
    # ["TRPO_Ablation1_1"],
    # ["TRPO_Ablation1_2"],
    # ["TRPO_Ablation1_3"],
    # ["TRPO_Ablation2_1"],
    # ["TRPO_Ablation2_2"],
    # ["TRPO_Ablation2_3"],
    # ["TRPO_Ablation5_1"],
    # ["TRPO_Ablation5_2"],
    # ["TRPO_Ablation5_3"],
    # ["PPO_neghrand_1"],
    # ["PPO_neghrand_2"],
    # ["PPO_neghrand_3"]
    # ["PPO_empty_space_ls_1"],
    # ["PPO_baseline_1"],
    ["PPO_Rebuttal_3_1"],
    ["PPO_Rebuttal_3_2"],
    ["PPO_Rebuttal_3_3"],
    ["PPO_Rebuttal_3_4"],
]

for file_name in file_name_list:
    if "PPO_plot" not in file_name[0] and "TRPO_plot" not in file_name[0]:
        log_dir = "logs"
        if "TRPO" in file_name[0]:
            log_dir = "trpo_logs"

        directory = "../"+log_dir+"/"+env+"/"+file_name[0]
        print("------------------------------------")
        print("Working on "+file_name[0]+" directory")
        reward_values = []
        searchString = "results"

        for filename in os.listdir(directory):
            # Check if the filename starts with "results"
            if filename.startswith(searchString):
                # Load the rewards
                results = np.load(directory + "/" + filename)
                # Find the maximum reward
                max_reward = np.max(results)
                # Append the maximum reward to the rewards list
                reward_values.append(max_reward)

        # Convert reward_values to numpy array
        reward_values_np = np.array(reward_values)
        print(reward_values_np.shape)

        # Save the rewards
        os.makedirs("../final_results/"+env, exist_ok=True)
        if len(file_name) > 1:
            np.save("../final_results/"+env+"/"+file_name[1]+".npy", reward_values_np)
        else:
            np.save("../final_results/"+env+"/"+file_name[0]+".npy", reward_values_np)
    else:
        print("------------------------------------")
        print("Working on "+file_name[0]+" directory")

        if env == "Ant-v5":
            env_proxy = "Ant"
        else:
            env_proxy = env
            
        file = "../base_job_output/"+env_proxy+"/"+file_name[1]+".txt"

        reward_values = []

        with open(file, "r") as f:
            lines = f.readlines()

        # Go through each line and extract the reward if it's a reward line
        for line in lines:
            if line.startswith("avg 3 return on policy"):
                match = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if match:
                    reward = float(match[-1])  # get the last number in the line
                    reward_values.append(reward)

        # Convert rewards to numpy array for easier math
        reward_values_np = np.array(reward_values)
        print("Rewards shape: ", reward_values_np.shape)

        # Save the rewards
        os.makedirs("../final_results/"+env, exist_ok=True)
        if len(file_name) > 2:
            np.save("../final_results/"+env+"/"+file_name[2]+".npy", reward_values_np)
        else:
            np.save("../final_results/"+env+"/"+file_name[0]+".npy", reward_values_np)