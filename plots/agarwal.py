import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from data_collection_config import args_ant, args_half_cheetah, args_walker2d, args_humanoid, args_swimmer, args_pendulum, args_bipedal_walker, args_lunarlander, args_hopper, args_fetch_reach, args_fetch_reach_dense, args_fetch_push, args_fetch_push_dense

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

env = "Ant-v5"
# env = "HalfCheetah-v5"
# env = "Hopper-v5"
# env = "Walker2d-v5"
# env = "Humanoid-v5"
# env = "Swimmer-v5"
# env = "Pendulum-v1"
# env = "BipedalWalker-v3"
# env = "LunarLander-v3"

def baseline_normalize(method_scores, baseline_scores, eps=1e-8):
    # scores: [num_envs, num_seeds]
    denom = baseline_scores.mean(axis=1, keepdims=True)
    return method_scores / (denom + eps)

env_names = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5"]  # <-- your list
seed_list = [0, 1, 2, 3]  # <-- your seeds

def load_curve(algo: str, env: str, seed: int) -> np.ndarray:
    """
    Return 1D array of episodic returns over training for a single run.
    You must implement this based on how you saved results.
    """
    # EXAMPLE A: npy file per run
    # path = Path("results") / algo / env / f"seed{seed}.npy"
    # return np.load(path)

    directory = "../final_results/"+env+"/"+algo+"_"+str(seed+1)
    results = np.load(directory + ".npy")
    return results

def last10pct_mean(curve: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 1 or len(curve) < 2:
        raise ValueError(f"Bad curve shape: {curve.shape}")
    k = max(1, int(np.ceil(0.10 * len(curve))))
    return float(np.mean(curve[-k:]))

def build_score_matrix(algo: str) -> np.ndarray:
    mat = np.zeros((len(env_names), len(seed_list)), dtype=float)
    for i, env in enumerate(env_names):
        for j, seed in enumerate(seed_list):
            curve = load_curve(algo, env, seed)   # <-- you implement
            mat[i, j] = last10pct_mean(curve)
    return mat

ppo_scores = build_score_matrix("PPO_normal_training")
explop_scores = build_score_matrix("PPO_upper_bound")
trpo_scores = build_score_matrix("TRPO_normal_training")
explot_scores = build_score_matrix("TRPO_upper_bound")

# Example: dict of [env, seed] arrays
scores = {
  "ExploRLer-P": explop_scores,
  "PPO": ppo_scores,
  "ExploRLer-T": explot_scores,
  "TRPO": trpo_scores,
}

assert ppo_scores.shape == explop_scores.shape
assert np.all(np.isfinite(ppo_scores))
assert np.all(np.isfinite(explop_scores))

norm_scores = {
  "ExploRLer-P": baseline_normalize(scores["ExploRLer-P"], scores["PPO"]),
  "PPO": baseline_normalize(scores["PPO"], scores["PPO"]),  # should be ~1
}

# IQM + 95% stratified bootstrap CI (Agarwal et al.)
aggregate_func = lambda x: metrics.aggregate_iqm(x)  # expects [env, seed]
point_est, ci = rly.get_interval_estimates(
    norm_scores,
    aggregate_func,
    reps=2000,
)

print("IQM:", point_est)
print("95% CI:", ci)

env_improvement = norm_scores["ExploRLer-P"].mean(axis=1)
for env, val in zip(env_names, env_improvement):
    print(env, f"{val:.2f}")

# ---------------------------------------------------------------------------------

print("---------------------------------------------------------------")

norm_scores = {
  "ExploRLer-T": baseline_normalize(scores["ExploRLer-T"], scores["TRPO"]),
  "TRPO": baseline_normalize(scores["TRPO"], scores["TRPO"]),  # should be ~1
}

# IQM + 95% stratified bootstrap CI (Agarwal et al.)
aggregate_func = lambda x: metrics.aggregate_iqm(x)  # expects [env, seed]
point_est, ci = rly.get_interval_estimates(
    norm_scores,
    aggregate_func,
    reps=2000,
)

print("IQM:", point_est)
print("95% CI:", ci)

env_improvement = norm_scores["ExploRLer-T"].mean(axis=1)
for env, val in zip(env_names, env_improvement):
    print(env, f"{val:.2f}")

# -----------------------------------------------------------------------------------

norm_scores = {
  "ExploRLer-P": baseline_normalize(scores["ExploRLer-P"], scores["PPO"]),
  "PPO": baseline_normalize(scores["PPO"], scores["PPO"]),  # should be ~1
  "ExploRLer-T": baseline_normalize(scores["ExploRLer-T"], scores["TRPO"]),
  "TRPO": baseline_normalize(scores["TRPO"], scores["TRPO"]),  # should be ~1
}

# Thresholds (tau)
tau = np.linspace(0.5, 2.5, 201)  # 201 points is fine

# IMPORTANT: plot_utils.plot_performance_profiles expects "score_distributions"
# computed by rly.create_performance_profile
score_distributions, score_distributions_cis = rly.create_performance_profile(
    norm_scores, tau
)

fig, ax = plt.subplots(ncols=1, figsize=(5.5, 4))
plot_utils.plot_performance_profiles(
    score_distributions,
    tau,
    performance_profile_cis=score_distributions_cis,  # optional but recommended
    xlabel="Normalized return threshold",
    ylabel="Fraction of runs",
    ax=ax,
)

ax.legend(
    ["PPO", "ExploRLer-P", "TRPO", "ExploRLer-T"],
    loc="upper right",
    frameon=False,
    fontsize=16
)

ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("performance_profile.pdf")