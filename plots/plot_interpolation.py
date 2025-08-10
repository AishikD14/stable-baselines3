import numpy as np
from scipy.interpolate import interp1d

def interpolate_to_length(arr, target_len):
    original_len = len(arr)
    original_x = np.linspace(0, 1, original_len)
    target_x = np.linspace(0, 1, target_len)
    interpolator = interp1d(original_x, arr, kind='linear')
    return interpolator(target_x)


def interpolate_after_fixed(arr, fixed_len, target_len):
    # Keep the first `fixed_len` points as-is
    fixed_part = arr[:fixed_len]
    
    remaining = arr[fixed_len:]
    remaining_len = len(remaining)
    target_remaining_len = target_len - fixed_len

    if remaining_len == 0:
        # Nothing to interpolate; just pad with last value
        interpolated = np.full(target_remaining_len, fixed_part[-1])
    else:
        # Interpolate only the remaining part
        x_old = np.linspace(0, 1, remaining_len)
        x_new = np.linspace(0, 1, target_remaining_len)
        interpolator = interp1d(x_old, remaining, kind='linear')
        interpolated = interpolator(x_new)
    
    return np.concatenate([fixed_part, interpolated])

# env = "Ant-v5"
# env = "HalfCheetah-v5"
env = "Hopper-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "Swimmer-v5"
# env = "Pendulum-v1"
# env = "BipedalWalker-v3"
# env = "LunarLander-v3"

# Load the arrays
r1 = np.load("../combined_results/"+env+"/PPO_upper_bound.npy")
# r2 = np.load("../combined_results/Ant-v5/PPO_Ablation4.npy")
r3 = np.load("../combined_results/"+env+"/PPO_normal_training.npy")

# Interpolate all to the same length
fixed_len = 1954
# target_len = max(len(r1), len(r2), len(r3))
target_len = len(r3)

if env in ["Pendulum-v1", "BipedalWalker-v3", "LunarLander-v3"]:
    r1_interp = interpolate_to_length(r1, target_len)
    # r2_interp = interpolate_to_length(r2, target_len)
else:
    r1_interp = interpolate_after_fixed(r1, fixed_len, target_len)
    # r2_interp = interpolate_after_fixed(r2, fixed_len, target_len)


print(r1_interp.shape)
# print(r2_interp.shape)

# Save the interpolated arrays
np.save("../combined_results/"+env+"/PPO_upper_bound_interpolated.npy", r1_interp)
# np.save("../combined_results/Ant-v5/PPO_Ablation4_interpolated.npy", r2_interp)
