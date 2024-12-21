import numpy as np
import matplotlib.pyplot as plt
import os

start_iteration = 1000

# end_iteration = 1010
# x_step_size = 1
# iterationString = ""

end_iteration = 1100
x_step_size = 10
iterationString = "_1"

x = np.arange(start_iteration+1, end_iteration+1, x_step_size)

# -------------------------------------------------------

# Load the rewards from the random sample directory

# Loop through the directories and load the rewards
directory = "../logs/Ant-v5/PPO_random_search"+iterationString
print("------------------------------------")
print("Working on random_sample directory")
rewards = []
try:
    times = np.load(directory + "/time.npy")
    times = list(times)

    time = [0]*len(times)
    time[0] = times[0]

    for i in range(1, len(times)):
        time[i] = times[i] - times[i-1]

    # Plot the rewards
    plt.plot(x, time, label="Random Sample")
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
    times = np.load(directory + "/time.npy")
    times = list(times)

    time = [0]*len(times)
    time[0] = times[0]

    for i in range(1, len(times)):
        time[i] = times[i] - times[i-1]

    time = time[::x_step_size]

    # Plot the rewards
    plt.plot(x, time, label="Neighbor Sampling+Empty Space")
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
    times = np.load(directory + "/time.npy")
    times = list(times)

    time = [0]*len(times)
    time[0] = times[0]

    for i in range(1, len(times)):
        time[i] = times[i] - times[i-1]

    time = time[::x_step_size]

    # Plot the rewards
    plt.plot(x, time, label="Neighbor Sampling+Random Walk")
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
    times = np.load(directory + "/time.npy")
    times = list(times)

    time = [0]*len(times)
    time[0] = times[0]

    for i in range(1, len(times)):
        time[i] = times[i] - times[i-1]

    time = time[::x_step_size]

    # Plot the rewards
    plt.plot(x, time, label="Random Sample+Empty Space")
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
    times = np.load(directory + "/time.npy")
    times = list(times)

    time = [0]*len(times)
    time[0] = times[0]

    for i in range(1, len(times)):
        time[i] = times[i] - times[i-1]

    time = time[::x_step_size]

    # Plot the rewards
    plt.plot(x, time, label="Random Sample+Random Walk")

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
    times = np.load(directory + "/time.npy")
    times = list(times)

    time = [0]*len(times)
    time[0] = times[0]

    for i in range(1, len(times)):
        time[i] = times[i] - times[i-1]

    time = time[::x_step_size]

    # Plot the rewards
    plt.plot(x, time, label="Normal Training")

except:
    print("Error in normal train directory")

# -------------------------------------------------------

plt.xlabel('Iteration')
plt.ylabel('Time Taken (s)')
# Customize the x-axis to show every value
plt.xticks(x)
# plt.yticks(np.arange(200, 700, 50))
# plt.ylim(200, 700)
plt.title('Time Plot')
plt.legend(fontsize='small')

# Check if images directory exists
if not os.path.exists('../images'):
    os.makedirs('../images')

if iterationString == "":
    plt.savefig('../images/time.png')
else:
    plt.savefig('../images/time'+iterationString+'.png')