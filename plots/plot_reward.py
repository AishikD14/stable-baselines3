import numpy as np
import matplotlib.pyplot as plt
import os

# start_iteration = 1000
# end_iteration = 1010
# x_step_size = 1
# iterationString = ""

start_iteration = 1000
end_iteration = 1100
x_step_size = 10
iterationString = "_2"

# start_iteration = 5000
# end_iteration = 5100
# x_step_size = 10
# iterationString = "_optimal_5M_1"

x = np.arange(start_iteration+1, end_iteration+1, x_step_size)

# -------------------------------------------------------

# Load the rewards from the random sample directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_random_search"+iterationString
print("------------------------------------")
print("Working on random_sample directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    # Plot the rewards
    # plt.plot(x, rewards, label="Random Sample")
except:
    print("Error in random sample directory")

# -------------------------------------------------------

# Load the rewards from the empty space directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_empty_space"+iterationString
print("------------------------------------")
print("Working on empty_space directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    rewards = rewards[:len(x)]

    # Plot the rewards
    plt.plot(x, rewards, label="Neighbor Sampling+Empty Space")
except :
    print("Error in empty space directory")

# -------------------------------------------------------

# Load the rewards from the nn_random_walk directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_neighbor_search_random_walk"+iterationString
print("------------------------------------")
print("Working on nn_random_walk directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    rewards = rewards[:len(x)]

    # Plot the rewards
    plt.plot(x, rewards, label="Neighbor Sampling+Random Walk")
except:
    print("Error in nn random walk directory")

# -------------------------------------------------------

# Load the rewards from the rsample_empty_space directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_random_search_empty_space"+iterationString
print("------------------------------------")
print("Working on rsample_empty_space directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    # Plot the rewards
    # plt.plot(x, rewards, label="Random Sample+Empty Space")
except:
    print("Error in rsample empty space directory")

# -------------------------------------------------------

# Load the rewards from the rsample_random_walk directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_random_search_random_walk"+iterationString
print("------------------------------------")
print("Working on rsample_random_walk directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    rewards = rewards[:len(x)]

    # Plot the rewards
    plt.plot(x, rewards, label="Random Sample+Random Walk")
except:
    print("Error in rsample random walk directory")

# -------------------------------------------------------

# Load the rewards from the normal_train directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_normal_train"+iterationString
print("------------------------------------")
print("Working on normal_train directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    rewards = rewards[:len(x)]

    # Plot the rewards
    plt.plot(x, rewards, label="Normal Training")
except:
    print("Error in normal train directory")

# -------------------------------------------------------

# Load the rewards from the PPO_empty_space_Annoy directory

# Loop through the directories and load the rewards
# directory = "../logs/Ant-v5/PPO_empty_space_Annoy"+iterationString
directory = "../logs/Ant-v5/PPO_empty_space_Annoy_2"
print("------------------------------------")
print("Working on PPO_empty_space_Annoy directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    # Plot the rewards
    # plt.plot(x, rewards, label="PPO_empty_space_Annoy")
except:
    print("Error in PPO_empty_space_Annoy directory")

# -------------------------------------------------------

# Load the rewards from the PPO_empty_space_Faiss directory

# Loop through the directories and load the rewards
# directory = "../logs/Ant-v5/PPO_empty_space_Faiss"+iterationString
directory = "../logs/Ant-v5/PPO_empty_space_Faiss_2"
print("------------------------------------")
print("Working on PPO_empty_space_Faiss directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    # Plot the rewards
    # plt.plot(x, rewards, label="PPO_empty_space_FAISS")
except:
    print("Error in PPO_empty_space_Faiss directory")

# -------------------------------------------------------

# Load the rewards from the random sample directory

# Loop through the directories and load the rewards
# directory = "../logs/Ant-v5/PPO_empty_space_hnswlib"+iterationString
directory = "../logs/Ant-v5/PPO_empty_space_hnswlib_2"
print("------------------------------------")
print("Working on PPO_empty_space_hnswlib directory")
rewards = []
try:
    for filename in os.listdir(directory):
        # Check if the filename starts with "results"
        if filename.startswith('results'):
            # print(filename)
            # Load the rewards
            results = np.load(directory + "/" + filename)
            # Find the maximum reward
            max_reward = np.max(results)
            # Append the maximum reward to the rewards list
            rewards.append(max_reward)

    # Choose appropriate value from the rewards
    rewards = rewards[::x_step_size]

    # Plot the rewards
    # plt.plot(x, rewards, label="PPO_empty_space_hnswlib")
except:
    print("Error in PPO_empty_space_hnswlib directory")

# -------------------------------------------------------

plt.xlabel('Samples (x200)')
plt.ylabel('Max Reward')
# Customize the x-axis to show every value
# plt.xticks(x)
# plt.yticks(np.arange(200, 700, 50))
# plt.ylim(200, 700)
plt.title('Reward Plot')
plt.legend(fontsize='small')

# Check if images directory exists
if not os.path.exists('../images'):
    os.makedirs('../images')

if iterationString == "":
    plt.savefig('../images/rewards.png')
else:
    plt.savefig('../images/rewards'+iterationString+'.png')