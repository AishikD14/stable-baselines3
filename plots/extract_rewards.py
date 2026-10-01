import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_ant_maze_dense, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker, args_lunarlander, args_hopper, args_fetch_reach, args_fetch_reach_dense, args_fetch_push, args_fetch_push_dense, args_point_maze_dense, args_metaworld_reach
import re

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

# env = "Ant-v5"
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
# env = "AntMaze_UMazeDense-v5"
# env = "PointMaze_UMazeDense-v3"
env = "MetaWorldReach-v0"

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
elif env == "Hopper-v5":
    args = args_hopper.get_args(rest_args)
elif env == "FetchReach-v4":
    args = args_fetch_reach.get_args(rest_args)
elif env == "FetchReachDense-v4":
    args = args_fetch_reach_dense.get_args(rest_args)
elif env == "FetchPush-v4":
    args = args_fetch_push.get_args(rest_args)
elif env == "FetchPushDense-v4":
    args = args_fetch_push_dense.get_args(rest_args)
elif env == "PointMaze_UMazeDense-v3":
    args = args_point_maze_dense.get_args(rest_args)
elif env == "AntMaze_UMazeDense-v5":
    args = args_ant_maze_dense.get_args(rest_args)
elif env == "MetaWorldReach-v0":
    args = args_metaworld_reach.get_args(rest_args)

# PointMaze, AntMaze, and MetaWorldReach keep dense return as the primary metric. Success is
# extracted only as an additional aligned reporting metric.
MAZE_SUCCESS_ENVS = {"PointMaze_UMazeDense-v3", "AntMaze_UMazeDense-v5","MetaWorldReach-v0"}

def metric_file_sort_key(filename):
    """Sort results/success files by their saved iteration range.

    This preserves the intended chronological extraction order instead of relying
    on the filesystem-dependent order returned by os.listdir().
    """
    match = re.match(r"^(?:results|success)_(-?\d+)_(-?\d+)\.npy$", filename)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), filename)
    return (1, filename)

# Pendulum-v1
file_name_list = [
    # ["PPO_plot_1", "ppo_plot_out", "PPO_pretrain_1"],
    # ["PPO_plot_2", "ppo_plot_1_out", "PPO_pretrain_2"],
    # ["PPO_plot_3", "ppo_plot_2_out", "PPO_pretrain_3"],
    # ["PPO_plot_4", "ppo_plot_3_out", "PPO_pretrain_4"],
    # ["PPO_plot_5", "ppo_pretrain_5_out", "PPO_pretrain_5"],
    # ["PPO_plot_6", "ppo_pretrain_6_out", "PPO_pretrain_6"],
    # ["PPO_plot_7", "ppo_pretrain_7_out", "PPO_pretrain_7"],
    # ["PPO_plot_8", "ppo_pretrain_8_out", "PPO_pretrain_8"],
    # ["PPO_plot_9", "ppo_pretrain_9_out", "PPO_pretrain_9"],
    # ["PPO_plot_10", "ppo_pretrain_10_out", "PPO_pretrain_10"],
    # ["TRPO_plot_1", "trpo_plot_out", "TRPO_pretrain_1"],
    # ["TRPO_plot_2", "trpo_plot_1_out", "TRPO_pretrain_2"],
    # ["TRPO_plot_3", "trpo_plot_2_out", "TRPO_pretrain_3"],
    # ["TRPO_plot_4", "trpo_plot_3_out", "TRPO_pretrain_4"],
    # ["TRPO_plot_5", "trpo_pretrain_5_out", "TRPO_pretrain_5"],
    # ["TRPO_plot_6", "trpo_pretrain_6_out", "TRPO_pretrain_6"],
    # ["TRPO_plot_7", "trpo_pretrain_7_out", "TRPO_pretrain_7"],
    # ["TRPO_plot_8", "trpo_pretrain_8_out", "TRPO_pretrain_8"],
    # ["TRPO_plot_9", "trpo_pretrain_9_out", "TRPO_pretrain_9"],
    # ["TRPO_plot_10", "trpo_pretrain_10_out", "TRPO_pretrain_10"],
    # ["SAC_plot_1", "sac_plot_1_out", "SAC_pretrain_1"],
    # ["SAC_plot_2", "sac_plot_2_out", "SAC_pretrain_2"],
    # ["SAC_plot_3", "sac_plot_3_out", "SAC_pretrain_3"],
    # ["SAC_plot_4", "sac_plot_4_out", "SAC_pretrain_4"],
    # ["PPO_normal_training_1"],
    # ["PPO_normal_training_2"],
    # ["PPO_normal_training_3"],
    # ["PPO_normal_training_4"],
    # ["PPO_normal_training_5"],
    # ["PPO_normal_training_6"],
    # ["PPO_normal_training_7"],
    # ["PPO_normal_training_8"],
    # ["PPO_normal_training_9"],
    # ["PPO_normal_training_10"],
    # ["SAC_normal_training_1"],
    # ["SAC_normal_training_2"],
    # ["SAC_normal_training_3"],
    # ["SAC_normal_training_4"],
    # ["PPO_normal_training_7", "PPO_normal_training_4"],
    ["PPO_pretrain_1"],
    ["PPO_pretrain_2"],
    ["PPO_pretrain_3"],
    ["PPO_pretrain_4"],
    # ["PPO_upper_bound_1"],
    # ["PPO_upper_bound_2"],
    # ["PPO_upper_bound_3"],
    # ["PPO_upper_bound_4"],
    # ["PPO_upper_bound_5"],
    # ["PPO_upper_bound_6"],
    # ["PPO_upper_bound_7"],
    # ["PPO_upper_bound_8"],
    # ["PPO_upper_bound_9"],
    # ["PPO_upper_bound_10"],
    # ["PPO_normal_train_60_1"],
    # ["SAC_upper_bound_1"],
    # ["SAC_upper_bound_2"],
    # ["SAC_upper_bound_3"],
    # ["SAC_upper_bound_4"],
    # ["TRPO_normal_training_1"],
    # ["TRPO_normal_training_2"],
    # ["TRPO_normal_training_3"],
    # ["TRPO_normal_training_4"],
    # ["TRPO_normal_training_5"],
    # ["TRPO_normal_training_6"],
    # ["TRPO_normal_training_7"],   
    # ["TRPO_normal_training_8"],
    # ["TRPO_normal_training_9"],
    # ["TRPO_normal_training_10"],
    # ["TRPO_upper_bound_1"],
    # ["TRPO_upper_bound_2"],
    # ["TRPO_upper_bound_3"],
    # ["TRPO_upper_bound_4"],
    # ["TRPO_upper_bound_5"],
    # ["TRPO_upper_bound_6"],
    # ["TRPO_upper_bound_7"],
    # ["TRPO_upper_bound_8"],
    # ["TRPO_upper_bound_9"],
    # ["TRPO_upper_bound_10"],
    # ["TRPO_normal_train_60_1"],
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
    # ["PPO_neghrand_3"],
    # ["PPO_neghrand_4"],
    # ["TRPO_neghrand_1"],
    # ["TRPO_neghrand_2"],
    # ["TRPO_neghrand_3"],
    # ["TRPO_neghrand_4"]
    # ["PPO_empty_space_ls_1"],
    # ["PPO_baseline_1"],
    # ["PPO_CheckpointAvg_1"],
    # ["PPO_CheckpointAvg_2"],
    # ["PPO_CheckpointAvg_3"],
    # ["PPO_CheckpointAvg_4"],
    # ["TRPO_CheckpointAvg_1"],
    # ["TRPO_CheckpointAvg_2"],
    # ["TRPO_CheckpointAvg_3"],
    # ["TRPO_CheckpointAvg_4"],
    # ["PPO_PBT_1"],
    # ["PPO_PBT_2"],
    # ["PPO_PBT_3"],
    # ["PPO_PBT_4"],
    # ["TRPO_PBT_1"],
    # ["TRPO_PBT_2"],
    # ["TRPO_PBT_3"],
    # ["TRPO_PBT_4"],
    # ["PPO_NoPretrain_1"],
    # ["PPO_NoPretrain_2"],
    # ["PPO_NoPretrain_3"],
    # ["PPO_NoPretrain_4"],
    # ["TRPO_NoPretrain_1"],
    # ["TRPO_NoPretrain_2"],
    # ["TRPO_NoPretrain_3"],
    # ["TRPO_NoPretrain_4"],
    # ["PPO_GuidedES_1"],
    # ["PPO_GuidedES_2"],
    # ["PPO_GuidedES_3"],
    # ["PPO_GuidedES_4"],
    # ["TRPO_GuidedES_1"],
    # ["TRPO_GuidedES_2"],
    # ["TRPO_GuidedES_3"],
    # ["TRPO_GuidedES_4"],
    # ["PPO_VFS_1"],
    # ["PPO_VFS_2"],
    # ["PPO_VFS_3"],
    # ["PPO_VFS_4"],
    # ["TRPO_VFS_1"],
    # ["TRPO_VFS_2"],
    # ["TRPO_VFS_3"],
    # ["TRPO_VFS_4"],
    # ["PPO_hyper_E_6_1"],
    # ["PPO_hyper_E_6_2"],
    # ["PPO_hyper_E_10_1"],
    # ["PPO_hyper_E_10_2"],
    # ["PPO_hyper_m_3_E_6_1"],
    # ["PPO_hyper_m_3_E_6_2"],
    # ["PPO_hyper_m_3_1"],
    # ["PPO_hyper_m_3_2"],
    # ["PPO_hyper_m_4_1"],
    # ["PPO_hyper_m_4_2"],
    # ["PPO_hyper_I_20_1"],
    # ["PPO_hyper_I_20_2"],
    # ["PPO_hyper_I_30_1"],
    # ["PPO_hyper_I_30_2"],
    # ["PPO_hyper_I_40_1"],
    # ["PPO_hyper_I_40_2"],
    # ["PPO_parallel_1"],
    # ["PPO_param_noise_1"],
    # ["PPO_param_noise_2"],
    # ["PPO_param_noise_3"],
    # ["PPO_param_noise_4"],
    # ["TRPO_param_noise_1"],
    # ["TRPO_param_noise_2"],
    # ["TRPO_param_noise_3"],
    # ["TRPO_param_noise_4"],
]

for file_name in file_name_list:
    if "PPO_plot" not in file_name[0] and "TRPO_plot" not in file_name[0] and "SAC_plot" not in file_name[0]:
        log_dir = "logs"
        if "TRPO" in file_name[0]:
            log_dir = "trpo_logs"
        elif "SAC" in file_name[0]:
            log_dir = "sac_logs"

        directory = "../"+log_dir+"/"+env+"/"+file_name[0]
        print("------------------------------------")
        print("Working on "+file_name[0]+" directory")
        reward_values = []
        success_values = []
        searchString = "results"

        if env in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
            searchString = "success"

        for filename in sorted(os.listdir(directory), key=metric_file_sort_key):
            # Keep the original metric-file selection logic unchanged.
            if filename.startswith(searchString):
                # Load the same metric as before.
                results = np.load(directory + "/" + filename)
                # Preserve the original max extraction behavior.
                max_reward = np.max(results)
                reward_values.append(max_reward)

                # For maze environments, also extract the success rate belonging to
                # the SAME max-return candidate. This adds reporting only and does not
                # change the return-based candidate-selection logic.
                if env in MAZE_SUCCESS_ENVS and searchString == "results":
                    success_filename = "success" + filename[len("results"):]
                    success_path = directory + "/" + success_filename

                    if not os.path.exists(success_path):
                        raise FileNotFoundError(
                            f"Missing matching maze success file for {filename}: {success_path}"
                        )

                    success_results = np.load(success_path)
                    flat_results = np.asarray(results).reshape(-1)
                    flat_success = np.asarray(success_results).reshape(-1)

                    if flat_results.size != flat_success.size:
                        raise ValueError(
                            f"Mismatched result/success sizes for {filename}: "
                            f"{flat_results.size} rewards vs {flat_success.size} successes"
                        )

                    # Match the training selector exactly. Maze policies are selected with
                    # np.argsort(cum_rews)[-1], so ties must use the same index convention.
                    best_idx = int(np.argsort(flat_results)[-1])
                    success_values.append(float(flat_success[best_idx]))

        # Convert reward_values to numpy array
        reward_values_np = np.array(reward_values)
        print(reward_values_np.shape)

        # Save rewards using the exact same output naming logic as before.
        os.makedirs("../final_results/"+env, exist_ok=True)
        if len(file_name) > 1:
            output_name = file_name[1]
        else:
            output_name = file_name[0]

        np.save("../final_results/"+env+"/"+output_name+".npy", reward_values_np)

        # Additional maze-only output; reward files remain unchanged.
        if env in MAZE_SUCCESS_ENVS:
            success_values_np = np.array(success_values)
            print("Success shape: ", success_values_np.shape)
            np.save("../final_results/"+env+"/"+output_name+"_success.npy", success_values_np)
    else:
        print("------------------------------------")
        print("Working on "+file_name[0]+" directory")

        if env == "Ant-v5":
            env_proxy = "Ant"
        else:
            env_proxy = env
            
        file = "../base_job_output/"+env_proxy+"/"+file_name[1]+".txt"

        reward_values = []
        success_values = []

        with open(file, "r") as f:
            lines = f.readlines()

        # Go through each line and extract the reward if it's a reward line
        for line in lines:
            if line.startswith("avg 3 return on policy"):
                # Existing environments keep the original parser. If maze success is
                # printed on the same line, exclude that field from reward parsing.
                reward_text = line
                if env in MAZE_SUCCESS_ENVS and "Success rate" in line:
                    reward_text = line.split("Success rate", 1)[0]

                match = re.findall(r"[-+]?\d*\.\d+|\d+", reward_text)
                if match:
                    reward = float(match[-1])
                    reward_values.append(reward)

                if env in MAZE_SUCCESS_ENVS and "Success rate" in line:
                    success_match = re.search(
                        r"Success rate\s*:?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
                        line,
                    )
                    if success_match:
                        success_values.append(float(success_match.group(1)))

            elif env in MAZE_SUCCESS_ENVS and line.startswith("Success rate"):
                success_match = re.search(
                    r"Success rate\s*:?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
                    line,
                )
                if success_match:
                    success_values.append(float(success_match.group(1)))

        # Convert rewards to numpy array for easier math
        reward_values_np = np.array(reward_values)
        print("Rewards shape: ", reward_values_np.shape)

        # Save rewards using the exact same output naming logic as before.
        os.makedirs("../final_results/"+env, exist_ok=True)
        if len(file_name) > 2:
            output_name = file_name[2]
        else:
            output_name = file_name[0]

        np.save("../final_results/"+env+"/"+output_name+".npy", reward_values_np)

        # Save maze success only when success values were actually present in the log.
        if env in MAZE_SUCCESS_ENVS and success_values:
            if len(success_values) != len(reward_values):
                raise ValueError(
                    f"Mismatched reward/success counts in {file}: "
                    f"{len(reward_values)} rewards vs {len(success_values)} successes"
                )
            success_values_np = np.array(success_values)
            print("Success shape: ", success_values_np.shape)
            np.save("../final_results/"+env+"/"+output_name+"_success.npy", success_values_np)
