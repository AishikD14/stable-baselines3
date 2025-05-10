import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_bipedal_walker, args_pendulum
import re

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "Swimmer-v5"
env = "Pendulum-v1"
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
elif env == "BipedalWalker-v3":
    args = args_bipedal_walker.get_args(rest_args)
elif env == "Pendulum-v1":
    args = args_pendulum.get_args(rest_args)

if not hasattr(args, 'n_envs'):
    args.n_envs = 1

START_ITER = 1000000 // (args.n_steps_per_rollout*args.n_envs)
SEARCH_INTERV = 1 # Since PPO make n_epochs=10 updates with each rollout, we can set this to 1 instead of 10
NUM_ITERS = 3000000 // (args.n_steps_per_rollout*args.n_envs)

# Ant-v5
file_name_list = [
    # ["PPO_normal_training_1"],
    # ["PPO_normal_training_2"],
    # ["PPO_normal_training_3"],
    # ["PPO_FQE_1", "ppo_fqe_1_out"],
    # ["PPO_FQE_2", "ppo_fqe_2_out"],
    # ["PPO_FQE_3"],
    ["TRPO_normal_training_1"],
    ["TRPO_normal_training_2"],
    ["TRPO_normal_training_3"],
    # ["PPO_upper_bound_1", "PPO_upper_bound_2"],
    # ["PPO_upper_bound_2"],
    # ["PPO_upper_bound_3"],
]

for file_name in file_name_list:
    # if "FQE" not in file_name[0]:
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

    # ----------------------------------------------------------------------

    # num_saves = (NUM_ITERS - START_ITER) // SEARCH_INTERV

    # # Define index space for interpolation
    # sparse_indices = np.linspace(0, 1, len(reward_values_np))
    # dense_indices = np.linspace(0, 1, num_saves)

    # # Perform linear interpolation
    # dense_rewards = np.interp(dense_indices, sparse_indices, reward_values_np)
    # print(dense_rewards.shape)

    # Save if needed
    # np.save("../final_results/"+env+"/"+file_name[0]+".npy", dense_rewards)

    # ----------------------------------------------------------------------

    # Determine block size and trim to make it divisible
    block_size = 97 // 48  
    trimmed_len = block_size * 48
    trimmed = reward_values_np[:trimmed_len]

    # Reshape and downsample
    downsampled_rewards = trimmed.reshape(48, block_size).mean(axis=1)
    print(downsampled_rewards.shape)

    # Save if needed
    np.save("../final_results/"+env+"/"+file_name[0]+".npy", downsampled_rewards)