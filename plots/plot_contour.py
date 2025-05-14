import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
import torch
from scipy.interpolate import griddata

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

def load_weights(arng, directory, env):
    policies = []

    for i in range(10):
        policy_vec = []

        # new_model = PPO("MlpPolicy", env, verbose=0, device='cpu')
        # new_model.load(f'logs/{directory}/models/agent{i}.zip', device='cpu')

        ckp = torch.load(f'../logs/{directory}/models/agent{i+1}.zip', map_location=torch.device('cpu'))

        # ckp = new_model.policy.state_dict()
        ckp_layers = ckp.keys()

        for layer in ckp_layers:
            if 'value_net' not in layer:
                policy_vec.append(ckp[layer].detach().numpy().reshape(-1))

        policy_vec = np.concatenate(policy_vec)
        policies.append(policy_vec)

    policies = np.array(policies)

    return policies


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

# plot_item = ["PPO_contour_1", "1953_1954"]
plot_item = ["PPO_contour_2", "2343_2344"]

reward_values = []

# -------------------------------------------------------

# Load sample points
directory = "../logs/"+env+"/"+plot_item[0]

points = np.load(directory + "/sample_points_"+plot_item[1]+".npy")

print("Sample points loaded")
print("Sample points shape: ", points.shape)

for filename in os.listdir(directory):
    if filename.startswith("results"):
        # Load the rewards
        results = np.load(directory + "/" + filename)

        # Convert to list
        results = results.tolist()

        reward_values.append(results)

print(len(reward_values))

# Take the first 1000 points
points = points[:len(reward_values)]
# points = points[:80]
# reward_values = reward_values[:80]

print("Sample points shape after slicing: ", points.shape)

# --------------------------------------------------------------------------------

# Load the original checkppoints
DIR = env+"/"+plot_item[0]
checkpoints = load_weights(10, DIR, env)

print("Checkpoints loaded")
print("Shape of checkpoints:", checkpoints.shape)

# # Add the checkpoints to the sample points
# points = np.concatenate((points, checkpoints), axis=0)

# print("Sample points shape after adding checkpoints: ", points.shape)

# ---------------------------------------------------------------------------------

# Do PCA on the sample points
pca = PCA(n_components=2)
pca.fit(checkpoints)
X_pca = pca.transform(points)
print("Sample points PCA shape: ", X_pca.shape)

# ----------------------------------------------------------------------------------

x = X_pca[:, 0]
y = X_pca[:, 1]
z = reward_values

# Define grid.
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)

# Step 3: Interpolate reward values on the grid
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Step 4: Plot contour map
plt.figure(figsize=(8, 6))
contour = plt.contourf(xi, yi, zi, levels=20, cmap='viridis')
plt.scatter(x, y, c=z, edgecolors='k', cmap='viridis')  # Optional: Show original points
plt.colorbar(contour, label='Reward')

# Add PCA transformed checkpoints
checkpoints_pca = pca.transform(checkpoints)
plt.scatter(checkpoints_pca[:, 0], checkpoints_pca[:, 1], c='red', s=50, label='Checkpoints', edgecolors='k')
plt.legend()
# plt.grid()

plt.title("Contour Plot of Reward Values in PCA-Reduced Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig('../images/contour.png')
