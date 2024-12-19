import numpy as np
import matplotlib.pyplot as plt
import os

start_iteration = 1000

end_iteration = 1010
x_step_size = 1
iterationString = ""

# end_iteration = 800
# x_step_size = 5
# iterationString = "_" + str(end_iteration)

# total_reward_length = (end_iteration - start_iteration) / 10

# Create x values as indices starting from 1
# x_val = np.arange(1, total_reward_length + 1, x_step_size)
# x = [300 + 10 * x for x in x_val]

x = np.arange(start_iteration+1, end_iteration+1, x_step_size)

# -------------------------------------------------------

# Load the rewards from the empty space directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_empty_space"+iterationString
print("------------------------------------")
print("Working on empty_space directory")
rewards = []
# try:
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
plt.plot(x, rewards, label="Neighbor Sampling+Empty Space")
# except :
#     print("Error in empty space directory")

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

    # Plot the rewards
    plt.plot(x, rewards, label="Normal Training")
except:
    print("Error in normal train directory")

# -------------------------------------------------------

plt.xlabel('Iteration')
plt.ylabel('Max Reward')
# Customize the x-axis to show every value
plt.xticks(x)
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