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

file_name_list = [
    ["PPO_Rebuttal_2_1", "ppo_rebuttal_2_1_out"],
    ["PPO_Rebuttal_2_2", "ppo_rebuttal_2_2_out"],
    ["PPO_Rebuttal_2_3", "ppo_rebuttal_2_3_out"],
    ["PPO_Rebuttal_2_4", "ppo_rebuttal_2_4_out"],
]

for file_name in file_name_list:
    directory = "../base_job_output"+"/"+env+"/"+file_name[1]+".txt"
    print("------------------------------------")
    print("Working on "+file_name[0]+" directory")
    reward_values = []

    with open(directory, 'r') as f:
        for line in f:
            match = re.search(r'best agent cum rewards:\s*([0-9.]+)', line)
            if match:
                reward_values.append(float(match.group(1)))

    # Convert reward_values to numpy array
    reward_values_np = np.array(reward_values)
    print(reward_values_np.shape)

    # Save the rewards
    os.makedirs("../final_results/"+env, exist_ok=True)
    np.save("../final_results/"+env+"/"+file_name[0]+".npy", reward_values_np)