import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import os
import seaborn as sns

def baseline_normalize(method_scores, baseline_scores, eps=1e-8):
    # scores: [num_envs, num_seeds]
    denom = baseline_scores.mean(axis=1, keepdims=True)
    return method_scores / (denom + eps)

# env_names = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5"]  # <-- your list
env_names = ["Ant-v5", "Hopper-v5", "Humanoid-v5", "Walker2d-v5"]
# seed_list = [0, 1, 2, 3]  # <-- your seeds
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # <-- your seeds
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "final_results"))


def result_path(algo: str, env: str, seed: int) -> str:
    return os.path.join(RESULTS_DIR, env, f"{algo}_{seed + 1}.npy")


def has_complete_results(algo: str) -> bool:
    return all(os.path.exists(result_path(algo, env, seed)) for env in env_names for seed in seed_list)

def load_curve(algo: str, env: str, seed: int) -> np.ndarray:
    """
    Return 1D array of episodic returns over training for a single run.
    You must implement this based on how you saved results.
    """
    # EXAMPLE A: npy file per run
    # path = Path("results") / algo / env / f"seed{seed}.npy"
    # return np.load(path)

    results = np.load(result_path(algo, env, seed))
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
            curve = load_curve(algo, env, seed)
            mat[i, j] = last10pct_mean(curve)
    return mat

available_algos = {
    "PPO_normal_training": has_complete_results("PPO_normal_training"),
    "PPO_upper_bound": has_complete_results("PPO_upper_bound"),
    "TRPO_normal_training": False,
    "TRPO_upper_bound": False
}

for algo, is_available in available_algos.items():
    status = "found" if is_available else "missing"
    print(f"{algo}: {status}")

scores = {}
if available_algos["PPO_normal_training"]:
    scores["PPO"] = build_score_matrix("PPO_normal_training")
if available_algos["PPO_upper_bound"]:
    scores["ExploRLer-P"] = build_score_matrix("PPO_upper_bound")
if available_algos["TRPO_normal_training"]:
    scores["TRPO"] = build_score_matrix("TRPO_normal_training")
if available_algos["TRPO_upper_bound"]:
    scores["ExploRLer-T"] = build_score_matrix("TRPO_upper_bound")

for name, score_matrix in scores.items():
    assert np.all(np.isfinite(score_matrix)), f"Non-finite values found in {name}"

aggregate_func = lambda x: metrics.aggregate_iqm(x)  # expects [env, seed]
plot_norm_scores = {}
plot_algorithms = []

if "PPO" in scores and "ExploRLer-P" in scores:
    assert scores["PPO"].shape == scores["ExploRLer-P"].shape
    norm_scores = {
        "ExploRLer-P": baseline_normalize(scores["ExploRLer-P"], scores["PPO"]),
        "PPO": baseline_normalize(scores["PPO"], scores["PPO"]),
    }
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

    plot_norm_scores.update(norm_scores)
    plot_algorithms.extend(["ExploRLer-P", "PPO"])
else:
    print("Skipping PPO comparison because PPO or ExploRLer-P results are incomplete.")

if "TRPO" in scores and "ExploRLer-T" in scores:
    if plot_norm_scores:
        print("---------------------------------------------------------------")

    assert scores["TRPO"].shape == scores["ExploRLer-T"].shape
    norm_scores = {
        "ExploRLer-T": baseline_normalize(scores["ExploRLer-T"], scores["TRPO"]),
        "TRPO": baseline_normalize(scores["TRPO"], scores["TRPO"]),
    }
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

    plot_norm_scores.update(norm_scores)
    plot_algorithms.extend(["ExploRLer-T", "TRPO"])
else:
    print("Skipping TRPO comparison because TRPO or ExploRLer-T results are incomplete.")

if plot_norm_scores:
    tau = np.linspace(0.5, 2.5, 201)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        plot_norm_scores, tau
    )

    fig, ax = plt.subplots(ncols=1, figsize=(5.5, 4))
    plot_utils.plot_performance_profiles(
        score_distributions,
        tau,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(plot_algorithms, sns.color_palette('colorblind', n_colors=len(plot_algorithms)))),
        xlabel="Normalized return threshold",
        ylabel="Fraction of runs",
        ax=ax,
    )

    ax.legend(loc="upper right", fontsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("performance_profile.pdf")
    print("Saved performance profile to performance_profile.pdf")
else:
    print("No complete algorithm pair found, so no performance profile was generated.")
