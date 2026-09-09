import copy
import gymnasium as gym
# import gymnasium_robotics
from stable_baselines3 import PPO
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from collections import OrderedDict
import torch
import time
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
from environments.make_env import make_env
import pandas as pd
# Optional legacy/custom FQE support. The d3rlpy FQE path below does not require this.
try:
    from stable_baselines3.common.fqe import FQE
except ImportError:
    FQE = None
import torch.nn as nn
import argparse
from data_collection_config import args_ant_dir, args_ant, args_hopper, args_half_cheetah, args_walker2d, args_humanoid, args_cartpole, args_mountain_car, args_pendulum, args_swimmer, args_fetch_reach, args_fetch_reach_dense, args_fetch_push, args_fetch_push_dense, args_breakout_no_frameskip
from stable_baselines3.common.vec_env import SubprocVecEnv
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import QLearningAlgoBase
from d3rlpy.base import LearnableConfig
from d3rlpy.constants import ActionSpace
import matplotlib.pyplot as plt
from d3rlpy.torch_utility import TorchMiniBatch, TorchObservation
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
# import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from stable_baselines3.common.policies import ActorCriticPolicy
import math

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------
# GPU configuration
# -------------------------------------------------------------------------------------------------
# Use CUDA automatically when it is available. The environment simulation (Gymnasium/MuJoCo)
# remains on CPU, while PPO and the high-dimensional empty-space search can execute on the GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------------------------------
# FQE runtime configuration
# -------------------------------------------------------------------------------------------------
# Defaults intentionally preserve the current experiment. Override these only
# for explicit speed/quality ablations.
FQE_N_STEPS = int(os.environ.get("FQE_N_STEPS", "10000"))
FQE_N_STEPS_PER_EPOCH = int(os.environ.get("FQE_N_STEPS_PER_EPOCH", "1000"))
FQE_BATCH_SIZE = int(os.environ.get("FQE_BATCH_SIZE", "100"))

# Native FQE target-network refresh interval. The historical/default value is
# 100, so the canonical rank-study estimator remains unchanged unless this is
# overridden explicitly. The convergence study below varies this value only in
# analysis-only shadow FQE fits.
NATIVE_FQE_TARGET_UPDATE_INTERVAL = int(
    os.environ.get("NATIVE_FQE_TARGET_UPDATE_INTERVAL", "100")
)
if NATIVE_FQE_TARGET_UPDATE_INTERVAL <= 0:
    raise ValueError("NATIVE_FQE_TARGET_UPDATE_INTERVAL must be > 0.")

# Final long-horizon FQE convergence / stability diagnostic (completed study).
#
# This diagnostic is now OPT-IN by default because the convergence question has
# already been tested and the new state-conditional support study below reuses a
# 50k/100 FQE estimator. Set FQE_CONVERGENCE_STUDY=1 to reproduce it.
# Each token is: <gradient_steps>:<target_update_interval>. When enabled it
# performs the final clean comparison:
#   1) 10k/100  -- unchanged canonical baseline
#   2) 50k/100  -- more training with the original/stable target schedule
#   3) 50k/10   -- same 50k training budget with the faster target schedule
#
# This isolates the two questions left by the previous convergence study:
#   * Does simply training the original target=100 estimator longer help?
#   * At the same 50k budget, is target=100 more stable/useful than target=10?
#
# All three fits use the SAME full frozen replay data, candidate policies,
# online returns, critic initialization RNG state, and sampled replay-index
# prefix. They are analysis-only and never participate in ESA/PPO policy
# selection.
FQE_CONVERGENCE_STUDY = os.environ.get("FQE_CONVERGENCE_STUDY", "0") == "1"
_FQE_CONVERGENCE_SPEC = os.environ.get(
    "FQE_CONVERGENCE_CONFIGS",
    "10000:100,50000:100,50000:10",
)
FQE_CONVERGENCE_CONFIGS = []
for _conv_token in _FQE_CONVERGENCE_SPEC.split(","):
    _conv_token = _conv_token.strip()
    if not _conv_token:
        continue
    try:
        _conv_steps_text, _conv_target_text = _conv_token.split(":", 1)
        _conv_steps = int(_conv_steps_text)
        _conv_target = int(_conv_target_text)
    except Exception as exc:
        raise ValueError(
            "FQE_CONVERGENCE_CONFIGS entries must have the form "
            "<gradient_steps>:<target_update_interval>, e.g. 10000:10."
        ) from exc
    if _conv_steps <= 0 or _conv_target <= 0:
        raise ValueError(
            "FQE convergence gradient steps and target-update intervals must "
            "both be positive."
        )
    FQE_CONVERGENCE_CONFIGS.append((_conv_steps, _conv_target))

if FQE_CONVERGENCE_STUDY:
    if not FQE_CONVERGENCE_CONFIGS:
        raise ValueError(
            "FQE_CONVERGENCE_CONFIGS must contain at least one configuration."
        )
    if len(FQE_CONVERGENCE_CONFIGS) != len(set(FQE_CONVERGENCE_CONFIGS)):
        raise ValueError(
            "FQE_CONVERGENCE_CONFIGS contains duplicate configurations: "
            f"{FQE_CONVERGENCE_CONFIGS}"
        )

# Presentation/file-I/O switches only; these do not alter the FQE Bellman objective.
FQE_SHOW_PROGRESS = os.environ.get("FQE_SHOW_PROGRESS", "0") == "1"
FQE_USE_FILE_LOGGER = os.environ.get("FQE_USE_FILE_LOGGER", "0") == "1"

# Replay-coverage ablation (analysis only).
# The online oracle, PPO trajectory, candidate generation, and canonical rank-study
# result remain based on the FULL corrected replay buffer. Additional recent windows
# are evaluated only to diagnose replay coverage / distribution shift.
# Disabled by default for the focused state-conditional support study so each
# outer iteration trains only on the full replay buffer. Set
# REPLAY_COVERAGE_ABLATION=1 to
# restore the previous multi-window diagnostic; this does not affect PPO/ESA.
REPLAY_COVERAGE_ABLATION = os.environ.get("REPLAY_COVERAGE_ABLATION", "0") == "1"
_REPLAY_COVERAGE_SPEC = os.environ.get(
    "REPLAY_COVERAGE_WINDOWS",
    "35000,50000,65000,80000,full",
)
REPLAY_COVERAGE_WINDOWS = []
for _coverage_token in _REPLAY_COVERAGE_SPEC.split(","):
    _coverage_token = _coverage_token.strip().lower()
    if not _coverage_token:
        continue
    if _coverage_token == "full":
        REPLAY_COVERAGE_WINDOWS.append(None)
    else:
        _coverage_value = int(_coverage_token)
        if _coverage_value <= 0:
            raise ValueError(
                "Replay coverage windows must be positive integers or 'full'."
            )
        REPLAY_COVERAGE_WINDOWS.append(_coverage_value)

if REPLAY_COVERAGE_ABLATION:
    if not REPLAY_COVERAGE_WINDOWS:
        raise ValueError("REPLAY_COVERAGE_WINDOWS must contain at least one window.")
    if None not in REPLAY_COVERAGE_WINDOWS:
        raise ValueError(
            "Replay coverage ablation must include 'full' so the original "
            "full-buffer FQE result remains canonical."
        )

    _coverage_labels = [
        "full" if window is None else str(int(window))
        for window in REPLAY_COVERAGE_WINDOWS
    ]
    if len(_coverage_labels) != len(set(_coverage_labels)):
        raise ValueError(
            "REPLAY_COVERAGE_WINDOWS contains duplicate coverage windows: "
            f"{_coverage_labels}"
        )

# Hybrid shortlist metrics (analysis only).
# For a future hybrid selector, FQE ranks all candidates, then only the top-k
# are evaluated online. These metrics measure whether the oracle-best policy
# survives that shortlist and what regret remains after choosing the best
# online return inside the shortlist.
_HYBRID_TOPK_SPEC = os.environ.get("HYBRID_TOPK_VALUES", "3,5,10")
HYBRID_TOPK_VALUES = []
for _hybrid_token in _HYBRID_TOPK_SPEC.split(","):
    _hybrid_token = _hybrid_token.strip()
    if not _hybrid_token:
        continue
    _hybrid_k = int(_hybrid_token)
    if _hybrid_k <= 0:
        raise ValueError("HYBRID_TOPK_VALUES must contain positive integers.")
    HYBRID_TOPK_VALUES.append(_hybrid_k)

if not HYBRID_TOPK_VALUES:
    raise ValueError("HYBRID_TOPK_VALUES must contain at least one k value.")
if len(HYBRID_TOPK_VALUES) != len(set(HYBRID_TOPK_VALUES)):
    raise ValueError(
        "HYBRID_TOPK_VALUES contains duplicate values: "
        f"{HYBRID_TOPK_VALUES}"
    )
HYBRID_TOPK_VALUES = sorted(HYBRID_TOPK_VALUES)

# FQE backend used by the rank-correlation study.
# "native_batched" evaluates all empty-space candidates together with one
# vectorized PyTorch FQE workload. "d3rlpy" keeps the previous implementation
# available as a validation/fallback path.
FQE_BACKEND = os.environ.get("FQE_BACKEND", "native_batched").strip().lower()
if FQE_BACKEND not in {"native_batched", "d3rlpy"}:
    raise ValueError(
        "FQE_BACKEND must be either 'native_batched' or 'd3rlpy'. "
        f"Got {FQE_BACKEND!r}."
    )

if FQE_CONVERGENCE_STUDY and FQE_BACKEND != "native_batched":
    raise ValueError(
        "FQE_CONVERGENCE_STUDY is implemented for the native_batched backend "
        "only. Set FQE_BACKEND=native_batched or FQE_CONVERGENCE_STUDY=0."
    )

# d3rlpy's default continuous vector critic is:
# concat(observation, action) -> 256 ReLU -> 256 ReLU -> scalar Q.
# Keep these fixed so native FQE matches the existing d3rlpy configuration.
NATIVE_FQE_HIDDEN_UNITS = (256, 256)
NATIVE_FQE_ACTION_CHUNK_SIZE = int(
    os.environ.get("NATIVE_FQE_ACTION_CHUNK_SIZE", "8192")
)
if NATIVE_FQE_ACTION_CHUNK_SIZE <= 0:
    raise ValueError("NATIVE_FQE_ACTION_CHUNK_SIZE must be > 0.")

# Support-penalized FQE scoring.
#
# The LCB experiment is disabled by default (B=1, beta=0), restoring ordinary
# native FQE as the value estimator. Candidate ranking is then augmented with
# an explicit behavioral-support penalty computed on frozen replay data:
#
#   Score_pen(pi) = V_FQE(pi)
#                   - lambda * E_(s,a)~D[ ||pi(s) - a||_2^2 ]
#
# The squared L2 norm is summed over action dimensions exactly as written above
# (it is NOT divided by action_dim). The replay action is the corrected,
# actually-executed environment action stored by FQE replay semantics version 2.
#
# FQE_ENSEMBLE_SIZE and FQE_LCB_BETA are kept only for reproducibility/backward
# compatibility. The support-penalized ranking always uses the ordinary FQE
# value (ensemble mean if B>1), never the LCB value.
FQE_ENSEMBLE_SIZE = int(os.environ.get("FQE_ENSEMBLE_SIZE", "1"))
FQE_LCB_BETA = float(os.environ.get("FQE_LCB_BETA", "0.0"))
FQE_SUPPORT_PENALTY_LAMBDA = float(
    os.environ.get("FQE_SUPPORT_PENALTY_LAMBDA", "1.0")
)

if FQE_ENSEMBLE_SIZE <= 0:
    raise ValueError("FQE_ENSEMBLE_SIZE must be > 0.")
if not np.isfinite(FQE_LCB_BETA) or FQE_LCB_BETA < 0.0:
    raise ValueError("FQE_LCB_BETA must be finite and >= 0.")
if (
    not np.isfinite(FQE_SUPPORT_PENALTY_LAMBDA)
    or FQE_SUPPORT_PENALTY_LAMBDA < 0.0
):
    raise ValueError(
        "FQE_SUPPORT_PENALTY_LAMBDA must be finite and >= 0."
    )

# State-conditional kNN support estimator / filter (analysis only).
#
# The previous paired-action penalty compared pi(s_i) only with the single
# replay action a_i. This study estimates LOCAL action support instead:
#
#   1) standardize replay observations feature-wise;
#   2) for a deterministic subset of replay states s_i, find the k nearest
#      OTHER replay states N_k(s_i);
#   3) compute the candidate's local action distance
#        d_i(pi) = min_{j in N_k(s_i)} ||pi(s_i) - a_j||_2^2;
#   4) calibrate a behavior-support threshold from the leave-one-out replay
#      quantity
#        d_i(behavior) = min_{j in N_k(s_i)} ||a_i - a_j||_2^2;
#   5) call a candidate state "unsupported" when d_i(pi) exceeds the chosen
#      percentile of the behavior distances.
#
# The FILTER keeps candidates whose unsupported-state fraction is below
# KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION. To keep the diagnostic well-defined if
# that absolute threshold is too strict, the most-supported candidates are
# added until KNN_SUPPORT_MIN_KEEP is reached. This fallback uses support only,
# never online return or FQE.
#
# The filtered candidates are then ranked by a 50k/target100 raw FQE estimator,
# chosen from the final convergence study as the primary stable FQE baseline.
# This entire block is diagnostic only: PPO/ESA still selects from cum_rews.
KNN_SUPPORT_STUDY = os.environ.get("KNN_SUPPORT_STUDY", "1") == "1"
KNN_SUPPORT_K = int(os.environ.get("KNN_SUPPORT_K", "20"))
KNN_SUPPORT_QUERY_COUNT = int(
    os.environ.get("KNN_SUPPORT_QUERY_COUNT", "4096")
)
KNN_SUPPORT_QUERY_CHUNK_SIZE = int(
    os.environ.get("KNN_SUPPORT_QUERY_CHUNK_SIZE", "128")
)
KNN_SUPPORT_BEHAVIOR_PERCENTILE = float(
    os.environ.get("KNN_SUPPORT_BEHAVIOR_PERCENTILE", "95.0")
)
KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION = float(
    os.environ.get("KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION", "0.20")
)
KNN_SUPPORT_MIN_KEEP = int(os.environ.get("KNN_SUPPORT_MIN_KEEP", "5"))
KNN_SUPPORT_FQE_N_STEPS = int(
    os.environ.get("KNN_SUPPORT_FQE_N_STEPS", "50000")
)
KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL = int(
    os.environ.get("KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL", "100")
)

if KNN_SUPPORT_K <= 0:
    raise ValueError("KNN_SUPPORT_K must be > 0.")
if KNN_SUPPORT_QUERY_COUNT <= 0:
    raise ValueError("KNN_SUPPORT_QUERY_COUNT must be > 0.")
if KNN_SUPPORT_QUERY_CHUNK_SIZE <= 0:
    raise ValueError("KNN_SUPPORT_QUERY_CHUNK_SIZE must be > 0.")
if not (0.0 < KNN_SUPPORT_BEHAVIOR_PERCENTILE < 100.0):
    raise ValueError(
        "KNN_SUPPORT_BEHAVIOR_PERCENTILE must lie strictly between 0 and 100."
    )
if not (
    np.isfinite(KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION)
    and 0.0 <= KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION <= 1.0
):
    raise ValueError(
        "KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION must be finite and in [0, 1]."
    )
if KNN_SUPPORT_MIN_KEEP <= 0:
    raise ValueError("KNN_SUPPORT_MIN_KEEP must be > 0.")
if KNN_SUPPORT_FQE_N_STEPS <= 0:
    raise ValueError("KNN_SUPPORT_FQE_N_STEPS must be > 0.")
if KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL <= 0:
    raise ValueError("KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL must be > 0.")
if KNN_SUPPORT_STUDY and FQE_BACKEND != "native_batched":
    raise ValueError(
        "KNN_SUPPORT_STUDY is implemented for the native_batched backend only. "
        "Set FQE_BACKEND=native_batched or KNN_SUPPORT_STUDY=0."
    )

# Objective-alignment diagnostic (analysis only).
#
# This study records BOTH the ordinary undiscounted online episodic return and
# the historical PPO-gamma-discounted online return on the exact same candidate
# trajectories. With time-conditioned finite-horizon FQE enabled, the
# undiscounted return is the objective-aligned target; the discounted return is
# retained only as a historical diagnostic.
#
# IMPORTANT: the original online selector remains based on the undiscounted
# return in ``cum_rews``. Discounted returns are diagnostic-only and never
# affect PPO training, ESA candidate generation, candidate selection, replay
# data, or the canonical support-penalized ranking study.
OBJECTIVE_MISMATCH_STUDY = os.environ.get(
    "OBJECTIVE_MISMATCH_STUDY", "1"
) == "1"

# Time-conditioned finite-horizon FQE (analysis/OPE only).
#
# This changes ONLY the native FQE evaluation objective. PPO still trains with
# args.gamma/model.gamma, ESA candidate generation is unchanged, online
# evaluation remains the ordinary undiscounted episodic return, and online
# selection still uses cum_rews.
#
# When enabled, native FQE targets the same finite-episode objective as the
# online oracle:
#   Q(s_t, a_t, t) = r_t + Q(s_{t+1}, pi(s_{t+1}), t+1)
# with OPE gamma = 1 and bootstrap = 0 at either a true terminal or an episode
# truncation. The critic receives one extra normalized time feature t / H; the
# PPO candidate policy itself still receives ONLY the original environment
# observation.
TIME_CONDITIONED_FINITE_HORIZON_FQE = os.environ.get(
    "TIME_CONDITIONED_FINITE_HORIZON_FQE", "1"
) == "1"

# 0 means infer H from Gymnasium's registered environment specification
# (env.spec.max_episode_steps). For Ant-v5 this resolves to 1000. A positive
# value can be supplied for custom environments that do not expose a registered
# TimeLimit horizon.
FQE_FINITE_HORIZON_OVERRIDE = int(
    os.environ.get("FQE_FINITE_HORIZON", "0")
)
if FQE_FINITE_HORIZON_OVERRIDE < 0:
    raise ValueError("FQE_FINITE_HORIZON must be 0 (auto) or a positive integer.")

if TIME_CONDITIONED_FINITE_HORIZON_FQE and FQE_BACKEND != "native_batched":
    raise ValueError(
        "Time-conditioned finite-horizon FQE is implemented for the "
        "native_batched backend only. Set FQE_BACKEND=native_batched or "
        "TIME_CONDITIONED_FINITE_HORIZON_FQE=0."
    )


# Analysis-only state-occupancy kNN diagnostic.
#
# This diagnostic uses the SAME deterministic online candidate trajectories that
# are already collected for the ground-truth selector. It never adds those
# trajectories to replay, never trains FQE on them, and never changes best_idx.
#
# For each online-visited candidate state, measure its k-th nearest-neighbor
# distance to the frozen replay state distribution. By default the state metric
# is time-aware: replay/candidate observations are standardized feature-wise and
# t/H is appended and standardized as one additional feature. This matches the
# finite-horizon evaluator's dependence on time without exposing online data to
# FQE.
#
# A replay-calibrated 95th-percentile leave-one-out kNN radius defines an
# "occupancy OOD" threshold. Candidate diagnostics then ask whether larger
# occupancy shift is associated with larger FQE ranking error.
STATE_OCCUPANCY_KNN_STUDY = (
    os.environ.get("STATE_OCCUPANCY_KNN_STUDY", "1") == "1"
)
STATE_OCCUPANCY_K = int(os.environ.get("STATE_OCCUPANCY_K", "20"))
STATE_OCCUPANCY_REPLAY_QUERY_COUNT = int(
    os.environ.get("STATE_OCCUPANCY_REPLAY_QUERY_COUNT", "4096")
)
STATE_OCCUPANCY_CANDIDATE_QUERY_COUNT = int(
    os.environ.get("STATE_OCCUPANCY_CANDIDATE_QUERY_COUNT", "512")
)
STATE_OCCUPANCY_QUERY_CHUNK_SIZE = int(
    os.environ.get("STATE_OCCUPANCY_QUERY_CHUNK_SIZE", "128")
)
STATE_OCCUPANCY_BEHAVIOR_PERCENTILE = float(
    os.environ.get("STATE_OCCUPANCY_BEHAVIOR_PERCENTILE", "95.0")
)
STATE_OCCUPANCY_INCLUDE_TIME = (
    os.environ.get("STATE_OCCUPANCY_INCLUDE_TIME", "1") == "1"
)
STATE_OCCUPANCY_FQE_N_STEPS = int(
    os.environ.get("STATE_OCCUPANCY_FQE_N_STEPS", "50000")
)
STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL = int(
    os.environ.get("STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL", "100")
)

if STATE_OCCUPANCY_K <= 0:
    raise ValueError("STATE_OCCUPANCY_K must be > 0.")
if STATE_OCCUPANCY_REPLAY_QUERY_COUNT <= 0:
    raise ValueError("STATE_OCCUPANCY_REPLAY_QUERY_COUNT must be > 0.")
if STATE_OCCUPANCY_CANDIDATE_QUERY_COUNT <= 0:
    raise ValueError("STATE_OCCUPANCY_CANDIDATE_QUERY_COUNT must be > 0.")
if STATE_OCCUPANCY_QUERY_CHUNK_SIZE <= 0:
    raise ValueError("STATE_OCCUPANCY_QUERY_CHUNK_SIZE must be > 0.")
if not (0.0 < STATE_OCCUPANCY_BEHAVIOR_PERCENTILE < 100.0):
    raise ValueError(
        "STATE_OCCUPANCY_BEHAVIOR_PERCENTILE must lie strictly between 0 and 100."
    )
if STATE_OCCUPANCY_FQE_N_STEPS <= 0:
    raise ValueError("STATE_OCCUPANCY_FQE_N_STEPS must be > 0.")
if STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL <= 0:
    raise ValueError(
        "STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL must be > 0."
    )
if STATE_OCCUPANCY_KNN_STUDY and FQE_BACKEND != "native_batched":
    raise ValueError(
        "STATE_OCCUPANCY_KNN_STUDY is implemented for native_batched FQE only. "
        "Set FQE_BACKEND=native_batched or STATE_OCCUPANCY_KNN_STUDY=0."
    )
if (
    STATE_OCCUPANCY_KNN_STUDY
    and STATE_OCCUPANCY_INCLUDE_TIME
    and not TIME_CONDITIONED_FINITE_HORIZON_FQE
):
    raise ValueError(
        "STATE_OCCUPANCY_INCLUDE_TIME=1 requires "
        "TIME_CONDITIONED_FINITE_HORIZON_FQE=1."
    )

# Time-resolved state-occupancy diagnostic (analysis only).
#
# This reuses the SAME read-only candidate trajectories already captured by
# STATE_OCCUPANCY_KNN_STUDY. No new environment steps are taken. The default
# normalized boundaries [0, .10, .25, .50, 1] resolve to the Ant-v5 windows:
#   [0,100), [100,250), [250,500), [500,1000)
# when H=1000.
#
# Each window is calibrated against replay states from the SAME timestep window.
# This prevents a late candidate state from being declared "supported" merely
# because it resembles an early replay state, and gives every phase its own
# behavior 95th-percentile kNN-radius threshold. Observation/time
# standardization is still fitted only on the full frozen replay data so the
# feature metric itself is consistent across windows.
#
# The diagnostic also records SIGNED FQE ranking bias:
#   online_rank - fqe_rank > 0  => FQE ranks the candidate too highly.
# This is important because the whole-trajectory study found that occupancy
# novelty can create directional FQE overvaluation even when absolute rank
# error alone is less sensitive.
TIME_RESOLVED_OCCUPANCY_STUDY = (
    os.environ.get("TIME_RESOLVED_OCCUPANCY_STUDY", "1") == "1"
)
_TIME_RESOLVED_OCCUPANCY_BOUNDARIES_SPEC = os.environ.get(
    "TIME_RESOLVED_OCCUPANCY_BOUNDARIES",
    "0,0.10,0.25,0.50,1.0",
)
TIME_RESOLVED_OCCUPANCY_BOUNDARIES = tuple(
    float(token.strip())
    for token in _TIME_RESOLVED_OCCUPANCY_BOUNDARIES_SPEC.split(",")
    if token.strip()
)
TIME_RESOLVED_OCCUPANCY_REPLAY_QUERY_COUNT_PER_WINDOW = int(
    os.environ.get(
        "TIME_RESOLVED_OCCUPANCY_REPLAY_QUERY_COUNT_PER_WINDOW",
        "1024",
    )
)
TIME_RESOLVED_OCCUPANCY_CANDIDATE_QUERY_COUNT_PER_WINDOW = int(
    os.environ.get(
        "TIME_RESOLVED_OCCUPANCY_CANDIDATE_QUERY_COUNT_PER_WINDOW",
        "128",
    )
)

if TIME_RESOLVED_OCCUPANCY_STUDY and not STATE_OCCUPANCY_KNN_STUDY:
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_STUDY reuses the exact online trajectories "
        "captured by STATE_OCCUPANCY_KNN_STUDY. Keep "
        "STATE_OCCUPANCY_KNN_STUDY=1 or disable the time-resolved study."
    )
if len(TIME_RESOLVED_OCCUPANCY_BOUNDARIES) < 2:
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_BOUNDARIES must contain at least two values."
    )
if not np.all(np.isfinite(TIME_RESOLVED_OCCUPANCY_BOUNDARIES)):
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_BOUNDARIES must contain finite values."
    )
if abs(TIME_RESOLVED_OCCUPANCY_BOUNDARIES[0]) > 1e-12:
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_BOUNDARIES must start at 0."
    )
if abs(TIME_RESOLVED_OCCUPANCY_BOUNDARIES[-1] - 1.0) > 1e-12:
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_BOUNDARIES must end at 1."
    )
if any(
    right <= left
    for left, right in zip(
        TIME_RESOLVED_OCCUPANCY_BOUNDARIES[:-1],
        TIME_RESOLVED_OCCUPANCY_BOUNDARIES[1:],
    )
):
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_BOUNDARIES must be strictly increasing."
    )
if (
    TIME_RESOLVED_OCCUPANCY_BOUNDARIES[0] < 0.0
    or TIME_RESOLVED_OCCUPANCY_BOUNDARIES[-1] > 1.0
):
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_BOUNDARIES must lie in [0, 1]."
    )
if TIME_RESOLVED_OCCUPANCY_REPLAY_QUERY_COUNT_PER_WINDOW <= 0:
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_REPLAY_QUERY_COUNT_PER_WINDOW must be > 0."
    )
if TIME_RESOLVED_OCCUPANCY_CANDIDATE_QUERY_COUNT_PER_WINDOW <= 0:
    raise ValueError(
        "TIME_RESOLVED_OCCUPANCY_CANDIDATE_QUERY_COUNT_PER_WINDOW must be > 0."
    )

# d3rlpy trains floor(n_steps / n_steps_per_epoch) complete epochs. Reject
# invalid/non-divisible overrides so a requested FQE update budget is never
# silently shortened (e.g. 2500 with 1000 would otherwise run only 2000 steps).
if FQE_N_STEPS <= 0:
    raise ValueError("FQE_N_STEPS must be > 0.")
if FQE_N_STEPS_PER_EPOCH <= 0:
    raise ValueError("FQE_N_STEPS_PER_EPOCH must be > 0.")
if FQE_BATCH_SIZE <= 0:
    raise ValueError("FQE_BATCH_SIZE must be > 0.")
if FQE_N_STEPS % FQE_N_STEPS_PER_EPOCH != 0:
    raise ValueError(
        "FQE_N_STEPS must be exactly divisible by FQE_N_STEPS_PER_EPOCH "
        "because d3rlpy otherwise truncates the requested training budget. "
        f"Got FQE_N_STEPS={FQE_N_STEPS}, "
        f"FQE_N_STEPS_PER_EPOCH={FQE_N_STEPS_PER_EPOCH}."
    )


def print_device_info():
    print("---------------------------------")
    print(f"PyTorch device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available; using CPU fallback.")


def resolve_fqe_finite_horizon(env_name):
    """Resolve the evaluation horizon without changing the environment itself."""
    if FQE_FINITE_HORIZON_OVERRIDE > 0:
        return int(FQE_FINITE_HORIZON_OVERRIDE)

    try:
        spec = gym.spec(env_name)
    except Exception as exc:
        raise RuntimeError(
            "Could not infer the finite-horizon FQE episode length from "
            f"Gymnasium for {env_name!r}. Set FQE_FINITE_HORIZON explicitly."
        ) from exc

    max_episode_steps = getattr(spec, "max_episode_steps", None)
    if max_episode_steps is None:
        raise RuntimeError(
            f"Gymnasium environment {env_name!r} does not expose "
            "spec.max_episode_steps. Set FQE_FINITE_HORIZON explicitly."
        )

    max_episode_steps = int(max_episode_steps)
    if max_episode_steps <= 0:
        raise RuntimeError(
            f"Invalid max_episode_steps={max_episode_steps} for {env_name!r}."
        )
    return max_episode_steps


def _append_normalized_fqe_time(observations, timesteps, horizon):
    """Append t/H to critic observations; candidate PPO policies never see it."""
    if observations.ndim != 2:
        raise ValueError(
            "Time-conditioned native FQE expects rank-2 flat observations, "
            f"got shape {tuple(observations.shape)}."
        )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(-1)
    if timesteps.ndim != 2 or timesteps.shape[1] != 1:
        raise ValueError(
            "FQE timesteps must have shape [batch, 1], got "
            f"{tuple(timesteps.shape)}."
        )
    if observations.shape[0] != timesteps.shape[0]:
        raise ValueError(
            "FQE observation/timestep length mismatch: "
            f"{observations.shape[0]} vs {timesteps.shape[0]}."
        )

    horizon_tensor = torch.as_tensor(
        float(horizon), dtype=observations.dtype, device=observations.device
    )
    normalized_time = timesteps.to(
        dtype=observations.dtype, device=observations.device
    ) / horizon_tensor
    return torch.cat((observations, normalized_time), dim=-1)


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    arr = np.array(*args)
    # Preserve the original float32 behavior of torch.FloatTensor while placing the tensor
    # directly on the selected compute device.
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


class TorchNearestNeighbors:
    """Exact KNN with the same Euclidean-distance semantics as sklearn NearestNeighbors.

    This class intentionally exposes fit()/kneighbors() so the empty-space search keeps the
    same control flow as the original implementation. On CUDA it moves the expensive
    high-dimensional distance calculations to the GPU.
    """

    def __init__(self, n_neighbors=6, compute_device=None):
        self.n_neighbors = n_neighbors
        self.device = compute_device if compute_device is not None else device
        self.samples = None

    def fit(self, samples):
        if torch.is_tensor(samples):
            self.samples = samples.to(device=self.device, dtype=torch.float32)
        else:
            self.samples = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
        return self

    @torch.no_grad()
    def kneighbors(self, coor):
        if self.samples is None:
            raise RuntimeError("TorchNearestNeighbors.fit must be called before kneighbors.")

        if torch.is_tensor(coor):
            queries = coor.to(device=self.device, dtype=torch.float32)
        else:
            queries = torch.as_tensor(coor, dtype=torch.float32, device=self.device)

        # sklearn's default metric here is ordinary Euclidean (not squared Euclidean).
        all_distances = torch.cdist(queries, self.samples, p=2)
        distances, indices = torch.topk(
            all_distances,
            k=self.n_neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        return distances, indices

class ANNAnnoy:
    def __init__(self, dimension, n_neighbors) -> None:
        from annoy import AnnoyIndex
        self.index = AnnoyIndex(dimension, 'euclidean')
        self.index.set_seed(42)
        self.n_neighbors = n_neighbors
        self.samples = None
        self._built = False

    def fit(self, samples):
        # Annoy requires all items to be added before build().
        samples = np.asarray(samples)
        for i, s in enumerate(samples):
            self.index.add_item(i, s)
        self.index.build(10)
        self._built = True
        self.samples = np.concatenate((self.samples, samples)) if self.samples is not None else samples
        return self

    def query(self, coor):
        if not self._built:
            raise RuntimeError("ANNAnnoy.fit must be called before query.")

        # Match the batched-query behavior of sklearn/FAISS/HNSW used elsewhere in the code.
        coor = np.asarray(coor)
        all_indices = []
        all_distances = []
        for row in coor:
            indices, distances = self.index.get_nns_by_vector(
                row, self.n_neighbors, include_distances=True
            )
            all_indices.append(indices)
            all_distances.append(distances)
        return all_indices, all_distances
    
class ANNFaiss:
    def __init__(self, dimension, n_neighbors) -> None:
        import faiss
        cpu_index = faiss.IndexFlatL2(dimension)

        # Use a FAISS GPU index when a CUDA-enabled FAISS build is installed. If faiss-cpu is
        # installed, keep the original CPU behavior without failing.
        self.gpu_resources = None
        self.index = cpu_index
        if (
            device.type == "cuda"
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "index_cpu_to_gpu")
        ):
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, device.index or 0, cpu_index
                )
            except Exception as exc:
                # FAISS GPU support is optional. Falling back to the original CPU index keeps
                # ANN_lib="Faiss" functional even with a mismatched FAISS/CUDA installation.
                print(f"FAISS GPU initialization failed; using CPU FAISS instead: {exc}")
                self.gpu_resources = None
                self.index = cpu_index

        self.n_neighbors = n_neighbors
        self.samples = None
    def fit(self, samples):
        self.index.add(samples.astype(np.float32))
    def query(self, coor):
        distances, indices = self.index.search(coor.astype(np.float32), self.n_neighbors)
        return indices, distances

class ANNHnswlib:
    def __init__(self, dimension, n_neighbors) -> None:
        import hnswlib
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.n_neighbors = n_neighbors
        self.samples = None
    def fit(self, samples):
        self.index.add_items(samples, np.arange(samples.shape[0]))
        self.index.set_ef(50)
    def query(self, coor):
        indices,  distances = self.index.knn_query(coor, self.n_neighbors)
        return indices, distances
    
def F(epsilon, sigma, d):
    return 6 * epsilon * (2 * (sigma / d)**13 - (sigma / d)**7) / sigma

def elastic(es, neighbors, D):
    # GPU/Torch implementation. It is algebraically identical to the loop below, but evaluates
    # all neighbor force vectors in one batched operation.
    if torch.is_tensor(es):
        if not torch.is_tensor(neighbors):
            neighbors = torch.as_tensor(neighbors, dtype=es.dtype, device=es.device)
        if not torch.is_tensor(D):
            D = torch.as_tensor(D, dtype=es.dtype, device=es.device)

        sigma = torch.mean(D)
        epsilon = 0.5  # 2D case
        safe_D = torch.clamp(D, min=0.001)
        f = F(epsilon, sigma, safe_D)
        vecs = f.unsqueeze(-1) * (es - neighbors) / safe_D.unsqueeze(-1)
        return torch.sum(vecs, dim=0, keepdim=True)

    # Original NumPy implementation retained for CPU/ANN compatibility.
    vecs = []
    # sigma = 0.05 * (1 + 0.2 * es.shape[1])
    # sigma = 0.55
    sigma = np.mean(D)
    epsilon = 0.5  # 2D case
    for n, d in zip(neighbors, D):
        d = d if d > 0.001 else 0.001
        f = F(epsilon, sigma, d)
        vecs.append(f * (es - n) / d)
    direction = np.sum(vecs, axis=0)
    return direction

# Search the empty space policies
def empty_center(data, coor, neighbor, use_ANN, use_momentum, movestep, numiter):
    # CUDA/Torch path. The update equations and snapshot schedule are unchanged from the
    # original NumPy implementation.
    if torch.is_tensor(coor):
        orig_coor = coor.clone()
        cum_mag = torch.tensor(0.0, dtype=coor.dtype, device=coor.device)
        gamma = 0.3  # discount factor
        momentum = torch.zeros_like(coor)
        es_configs = []

        for i in range(numiter):
            if not use_ANN:
                # Exact KNN on the selected torch device (CUDA when available).
                distances_, adjs_ = neighbor.kneighbors(coor)
            else:
                # Existing ANN libraries expose NumPy interfaces. Keep their behavior, then
                # move only the returned neighbor metadata back to the tensor device.
                coor_np = coor.detach().cpu().numpy()
                adjs_np, distances_np = neighbor.query(coor_np)
                adjs_ = torch.as_tensor(np.array(adjs_np), dtype=torch.long, device=coor.device)
                distances_ = torch.as_tensor(np.array(distances_np), dtype=coor.dtype, device=coor.device)

            if i % 20 == 0:
                if use_momentum:
                    # Equivalent to es_configs.extend(coor.tolist()) for coor shape (1, D).
                    es_configs.append(coor.clone())

            direction = elastic(coor, data[adjs_[0]], distances_[0])
            mag = torch.linalg.vector_norm(direction)
            if mag.item() < 1e-7:
                break

            direction = direction / mag
            if use_momentum:
                direction = (
                    direction * mag / (cum_mag + mag)
                    + momentum * cum_mag / (cum_mag + mag)
                )

            # Keep the original in-place movement semantics.
            coor += direction * movestep

            if use_momentum:
                cum_mag = gamma * cum_mag + mag
                momentum = gamma * momentum + direction
                momentum = momentum / torch.linalg.vector_norm(momentum)

        es_configs.append(coor.clone())

        distance = torch.linalg.vector_norm(coor - orig_coor).item()
        if not use_momentum:
            return distance, coor
        return distance, torch.cat(es_configs, dim=0)

    # Original NumPy implementation retained unchanged for the CPU/ANN paths.
    orig_coor = coor.copy()
    cum_mag = 0
    gamma = 0.3 # discount factor
    momentum = np.zeros(coor.shape)
    es_configs = []
    for i in range(numiter):
        
        if not use_ANN:
            # Calculate the nearest neighbors of the agents using KNN
            distances_, adjs_ = neighbor.kneighbors(coor)

        else:
            # Calculate the nearest neighbors of the agents using Approximate Nearest Neighbors
            adjs_, distances_ = neighbor.query(coor)
            adjs_ = np.array(adjs_)

        if i % 20 == 0:
            if use_momentum:
                es_configs.extend(coor.tolist())

        direction = elastic(coor, data[adjs_[0]], distances_[0])
        mag = np.linalg.norm(direction)
        if mag < 1e-7:
            break
        direction /= mag
        if use_momentum:
            direction = direction * mag / (cum_mag + mag) + momentum * cum_mag / (cum_mag + mag)
        coor += direction * movestep

        if use_momentum:
            cum_mag = gamma * cum_mag + mag
            momentum = gamma * momentum + direction
            momentum /= np.linalg.norm(momentum)

    es_configs.extend(coor.tolist())
        
    if not use_momentum:
        return np.linalg.norm(coor - orig_coor), coor
    else:
        return np.linalg.norm(coor - orig_coor), np.array(es_configs)

def random_walk(data, coor, neighbor, use_momentum, movestep, numiter):
    use_momentum = False
    orig_coor = coor.copy()
    cum_mag = 0
    gamma = 0.3  # discount factor
    momentum = np.zeros(coor.shape)
    rw_configs = []

    for i in range(numiter):
        # Generate a random direction
        direction = np.random.uniform(-1, 1, coor.shape)
        
        # Normalize the random direction
        mag = np.linalg.norm(direction)
        if mag < 1e-7:
            break
        direction /= mag
        
        if use_momentum:
            direction = direction * mag / (cum_mag + mag) + momentum * cum_mag / (cum_mag + mag)

        coor += direction * movestep
        
        if use_momentum:
            cum_mag = gamma * cum_mag + mag
            momentum = gamma * momentum + direction
            momentum /= np.linalg.norm(momentum)
        
        # Store configurations periodically for elastic search purposes
        if i % 20 == 0:
            # if use_momentum:
            rw_configs.extend(coor.tolist())

    rw_configs.extend(coor.tolist())

    # if not use_momentum:
    #     return np.linalg.norm(coor - orig_coor), coor
    # else:
    return np.linalg.norm(coor - orig_coor), np.array(rw_configs)

def load_weights(arng, directory, env, saved_agents=False, seed=0, as_tensor=False):
    """Load the same policy parameters as before.

    as_tensor=False preserves the original NumPy return type for the alternative search modes.
    as_tensor=True keeps the flattened vectors on the selected torch device for GPU ESA/KNN.
    """
    policies = []
    target_device = device if as_tensor else torch.device("cpu")

    for i in range(10):
        policy_vec = []

        if saved_agents:
            ckp = torch.load(
                f'logs/{directory[:-2]}_{seed+1}/models/agent{i+1}.zip',
                map_location=target_device
            )
        else:
            ckp = torch.load(
                f'logs/{directory}/models/agent{i+1}.zip',
                map_location=target_device
            )

        ckp_layers = ckp.keys()

        for layer in ckp_layers:
            if 'value_net' not in layer:
                tensor = ckp[layer].detach().reshape(-1)
                if as_tensor:
                    policy_vec.append(tensor.to(device=device, dtype=torch.float32))
                else:
                    policy_vec.append(tensor.cpu().numpy())

        if as_tensor:
            policy_vec = torch.cat(policy_vec)
        else:
            policy_vec = np.concatenate(policy_vec)
        policies.append(policy_vec)

    if as_tensor:
        policies = torch.stack(policies, dim=0)
    else:
        policies = np.array(policies)

    return policies

def dump_weights(agent_net, es_models):
    policies = []
    for i in range(es_models.shape[0]):
        policy = OrderedDict()
        pivot = 0
        for layer in agent_net:
            if 'value_net' in layer:
                policy[layer] = agent_net[layer]
            else:
                sp = agent_net[layer].reshape(-1).shape[0]
                shape = agent_net[layer].shape
                values = es_models[i][pivot : pivot + sp]

                if torch.is_tensor(values):
                    policy[layer] = values.reshape(shape).to(
                        device=agent_net[layer].device,
                        dtype=agent_net[layer].dtype,
                    ).clone()
                else:
                    policy[layer] = FloatTensor(values.reshape(shape)).to(
                        device=agent_net[layer].device,
                        dtype=agent_net[layer].dtype,
                    )
                pivot += sp
        policies.append(policy)
    return policies


def state_dict_to_cpu(state_dict):
    """Return a CPU copy of a policy state dict without changing the in-memory policy."""
    return OrderedDict(
        (key, value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value))
        for key, value in state_dict.items()
    )


def agents_to_cpu(agents):
    """CPU copies used only for persistence/process transfer, matching the original file behavior."""
    return [state_dict_to_cpu(agent) for agent in agents]


def close_env_safely(env):
    """Close temporary evaluation environments without affecting experiment logic."""
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception as exc:
            print(f"Warning: temporary environment close failed: {exc}")

# Worker function for parallel empty center search
def empty_center_worker(args):
    dt, p, neigh, use_ANN, movestep, numiter = args
    dist, es_pol = empty_center(
        dt,
        p.reshape(1, -1),
        neigh,
        use_ANN,
        use_momentum=True,
        movestep=movestep,
        numiter=numiter
    )
    return es_pol

# Parallel empty center search
def parallel_empty_centers(points, dt, neigh, use_ANN, movestep=0.001, numiter=60):
    print("Running parallel empty center searches for", len(points), "points...")
    job_args = [
        (dt, p, neigh, use_ANN, movestep, numiter)
        for p in points
    ]

    n_workers = min(len(points), cpu_count())

    with Pool(processes=n_workers) as pool:
        results = pool.map(empty_center_worker, job_args)

    # results is a list of numpy arrays shaped (1, D)
    return np.concatenate(results, axis=0)

# Nearest neighbor search plus empty space search
def search_empty_space_policies(algo, directory, start, end, env, use_ANN, ANN_lib, saved_agents, agent_num=10, seed=0):
    print("---------------------------------")
    print("Searching empty space policies")

    parallel_esa = False

    # The default exact-KNN ESA path is the best fit for CUDA because both the policy vectors
    # and Euclidean neighbor calculations can remain as tensors. Existing ANN modes retain
    # their original NumPy interfaces and behavior.
    gpu_esa = device.type == "cuda" and not use_ANN and not parallel_esa
    if gpu_esa:
        print("Empty-space search backend: CUDA / PyTorch")
    else:
        print("Empty-space search backend: CPU/legacy path")

    dt = load_weights(
        range(start, end), directory, env, saved_agents, seed=seed, as_tensor=gpu_esa
    )
    print(tuple(dt.shape) if torch.is_tensor(dt) else dt.shape)

    if not use_ANN:
        # Calculate the nearest neighbors of the agents using exact KNN. On CUDA,
        # TorchNearestNeighbors has the same fit/kneighbors control flow as sklearn.
        if gpu_esa:
            neigh = TorchNearestNeighbors(n_neighbors=6, compute_device=device)
            neigh.fit(dt)
            _, adjs = neigh.kneighbors(dt[-agent_num:])
        else:
            neigh = NearestNeighbors(n_neighbors=6)
            neigh.fit(dt)
            _, adjs = neigh.kneighbors(dt[-agent_num:])

    else:
        # Calculate the nearest neighbors of the agents using Approximate Nearest Neighbors
        if ANN_lib == "Annoy":
            neigh = ANNAnnoy(dimension=dt.shape[1], n_neighbors=6)
        elif ANN_lib == "Faiss":
            neigh = ANNFaiss(dimension=dt.shape[1], n_neighbors=6)
        elif ANN_lib == "Hnswlib":
            neigh = ANNHnswlib(dimension=dt.shape[1], n_neighbors=6)
        neigh.fit(dt)
        adjs, _ = neigh.query(dt[-agent_num:])
        adjs = np.array(adjs)

    points = dt[adjs[:, 1:]]
    if torch.is_tensor(points):
        points = points.mean(dim=1)
    else:
        points = points.mean(axis=1)

    # Choose a subset of points
    # points = points[::4] #m=3
    # points = points[::3] #m=4
    points = points[::2] #m=5 (Base Version)

    # ---------------------------------------------------------------------------------------------------------

    # Non-parallel empty center search
    # CUDA already evaluates the vector arithmetic in parallel, so the original process-level ESA
    # switch remains disabled by default.
    if not parallel_esa:
        policies = []
        for p in points:
            a = empty_center(
                dt, p.reshape(1, -1), neigh, use_ANN,
                use_momentum=True, movestep=0.001, numiter=60
            )
            policies.append(a[1])

        if gpu_esa:
            policies = torch.cat(policies, dim=0)
        else:
            policies = np.concatenate(policies)
        print(tuple(policies.shape) if torch.is_tensor(policies) else policies.shape)

    # Parallel empty center search
    else:
        # Preserve the original multiprocessing implementation. When parallel_esa=True,
        # gpu_esa is disabled above so this branch receives NumPy/scikit-learn objects exactly
        # like the CPU version rather than spawning multiple CUDA contexts.
        policies = parallel_empty_centers(
            points=points,
            dt=dt,
            neigh=neigh,
            use_ANN=use_ANN,
            movestep=0.001,
            numiter=60
        )
        print(policies.shape)

    # ---------------------------------------------------------------------------------------------------------

    agents = dump_weights(algo.policy.state_dict(), policies)

    if not use_ANN:
        # Calculate the nearest neighbors of the generated agents using the same backend.
        if gpu_esa:
            neigh = TorchNearestNeighbors(n_neighbors=6, compute_device=device)
            neigh.fit(policies)
            _, adjs = neigh.kneighbors(policies)
        else:
            neigh = NearestNeighbors(n_neighbors=6)
            neigh.fit(policies)
            _, adjs = neigh.kneighbors(policies)

    else:
        # Calculate the nearest neighbors of the agents using Approximate Nearest Neighbors
        if ANN_lib == "Annoy":
            neigh = ANNAnnoy(dimension=policies.shape[1], n_neighbors=6)
        elif ANN_lib == "Faiss":
            neigh = ANNFaiss(dimension=policies.shape[1], n_neighbors=6)
        elif ANN_lib == "Hnswlib":
            neigh = ANNHnswlib(dimension=policies.shape[1], n_neighbors=6)
        neigh.fit(policies)
        adjs, _ = neigh.query(policies)
        adjs = np.array(adjs)

    points = policies[adjs[:, 1:]]
    if torch.is_tensor(points):
        points = points.mean(dim=1)
        print(tuple(points.shape))

        # Same Euclidean distance calculation as scipy.spatial.distance.euclidean, batched on GPU.
        distances = torch.linalg.vector_norm(policies - points, dim=1)
        average_distance = distances.mean().item()
    else:
        points = points.mean(axis=1)
        print(points.shape)

        # Original CPU distance calculation.
        distances = [euclidean(policy, point) for policy, point in zip(policies, points)]
        average_distance = np.mean(distances)

    print("Average distance of agents to nearest neighbors:", average_distance)

    return agents, average_distance

# Function to calculate the mean and covariance of the training data
def fit_gaussian_model(data):
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)
    return mean, covariance

#  Randomly sample from a Gaussian distribution of points
def random_search_policies(algo, directory, start, end, env, agent_num=10):
    dt = load_weights(range(start, end), directory, env)
    print(dt.shape)

    # Fit the Gaussian model to the training data with MLE
    mean, covariance = fit_gaussian_model(dt)
    # print("Mean of the Gaussian model:", mean)
    # print("Covariance of the Gaussian model:", covariance)

    # Sample from the fitted Gaussian distribution
    policies = np.random.multivariate_normal(mean, covariance, agent_num)
    print("Shape of generated policies:", policies.shape)
    
    # Calculate log-likelihood of the training data under the fitted Gaussian model
    # log_likelihood = np.sum(multivariate_normal.logpdf(dt, mean=mean, cov=covariance))
    # print("Log-likelihood of the training data under the fitted Gaussian model:", log_likelihood)

    agents = dump_weights(algo.policy.state_dict(), policies)

    # Distance Calculation 1

    # # Calculate the mean of the training agents
    # training_agents_mean = np.mean(dt, axis=0)

    # # Calculate the distance of each random agent to the mean of the training agents
    # distances = [euclidean(policy, training_agents_mean) for policy in policies]
    
    # # Average distance
    # average_distance = np.mean(distances)
    # print("Average distance of random agents to training agents:", average_distance)

    # ---------------------------------------------------------------------------------

    # Distance Calculation 2

    # Calculate the nearest neighbors of the random agents
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(policies)

    _, adjs = neigh.kneighbors(policies)
    points = policies[adjs[:, 1:]]
    points = points.mean(axis=1)
    print(points.shape)

    # Calculate the distance of each random agent to the mean of the nearest neighbors
    distances = [euclidean(policy, point) for policy, point in zip(policies, points)]

    # Average distance
    average_distance = np.mean(distances)
    print("Average distance of random agents to nearest neighbors:", average_distance)

    
    return agents, average_distance

# Neighbor search plus random walk
def neighbor_search_random_walk(algo, directory, start, end, env, saved_agents=False, agent_num=10):
    print("---------------------------------")
    print("Searching random policies")

    dt = load_weights(range(start, end), directory, env, saved_agents)
    print(dt.shape)

    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(dt)
    _, adjs = neigh.kneighbors(dt[-agent_num:])

    points = dt[adjs[:, 1:]]
    points = points.mean(axis=1)

    # Choose every second point
    points = points[::2]

    policies = []
    print(len(points))
    for p in points:
        a = random_walk(dt, p.reshape(1, -1), neigh, use_momentum=True, movestep=0.001, numiter=60)
        policies.append(a[1])
    policies = np.concatenate(policies)
    print(policies.shape)

    agents = dump_weights(algo.policy.state_dict(), policies)

    # Calculate the nearest neighbors of the agents
    # neigh = NearestNeighbors(n_neighbors=6)
    # neigh.fit(policies)
    # _, adjs = neigh.kneighbors(policies)
    
    # points = policies[adjs[:, 1:]]
    # points = points.mean(axis=1)
    # print(points.shape)

    # # Calculate the distance of each agent to the mean of the nearest neighbors
    # distances = [euclidean(policy, point) for policy, point in zip(policies, points)]

    # # Average distance
    # average_distance = np.mean(distances)
    # print("Average distance of agents to nearest neighbors:", average_distance)
    
    return agents, 0

# Random Sampling plus empty space search
def random_search_empty_space_policies(algo, directory, start, end, env, agent_num=10):
    dt = load_weights(range(start, end), directory, env)
    print(dt.shape)
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(dt)

    # Fit the Gaussian model to the training data with MLE
    mean, covariance = fit_gaussian_model(dt)
    # print("Mean of the Gaussian model:", mean)
    # print("Covariance of the Gaussian model:", covariance)

    # Sample from the fitted Gaussian distribution
    points = np.random.multivariate_normal(mean, covariance, agent_num)
    print("Shape of generated points:", points.shape)

    # print("done")

    policies = []
    print(len(points))
    for p in points:
        a = empty_center(
            dt, p.reshape(1, -1), neigh, use_ANN=False,
            use_momentum=True, movestep=0.001, numiter=400
        )
        policies.append(a[1])
    policies = np.concatenate(policies)
    print(policies.shape)

    agents = dump_weights(algo.policy.state_dict(), policies)

    # Calculate the nearest neighbors of the random agents
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(policies)

    _, adjs = neigh.kneighbors(policies)
    points = policies[adjs[:, 1:]]
    points = points.mean(axis=1)
    print(points.shape)

    # Calculate the distance of each random agent to the mean of the nearest neighbors
    distances = [euclidean(policy, point) for policy, point in zip(policies, points)]

    # Average distance
    average_distance = np.mean(distances)
    print("Average distance of agents to nearest neighbors:", average_distance)
    
    return agents, average_distance

# Random Sampling plus random walk
def random_search_random_walk(algo, directory, start, end, env, agent_num=10):
    dt = load_weights(range(start, end), directory, env)
    print(dt.shape)
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(dt)

    # Fit the Gaussian model to the training data with MLE
    mean, covariance = fit_gaussian_model(dt)
    # print("Mean of the Gaussian model:", mean)
    # print("Covariance of the Gaussian model:", covariance)

    # Sample from the fitted Gaussian distribution
    points = np.random.multivariate_normal(mean, covariance, agent_num)
    print("Shape of generated points:", points.shape)

    print("done")

    policies = []
    print(len(points))
    for p in points:
        a = random_walk(dt, p.reshape(1, -1), neigh, use_momentum=True, movestep=0.001, numiter=400)
        policies.append(a[1])
    policies = np.concatenate(policies)
    print(policies.shape)

    agents = dump_weights(algo.policy.state_dict(), policies)

    # Calculate the nearest neighbors of the random agents
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(policies)

    _, adjs = neigh.kneighbors(policies)
    points = policies[adjs[:, 1:]]
    points = points.mean(axis=1)
    print(points.shape)

    # Calculate the distance of each random agent to the mean of the nearest neighbors
    distances = [euclidean(policy, point) for policy, point in zip(policies, points)]

    # Average distance
    average_distance = np.mean(distances)
    print("Average distance of agents to nearest neighbors:", average_distance)
    
    return agents, average_distance

def load_state_dict(algo, params):
    algo.policy.load_state_dict(params)
    algo.policy.optimizer = algo.policy.optimizer_class(algo.policy.parameters(), lr=algo.learning_rate)
    algo.policy.to(device)

def evaluation_callback(localvars, globalvars):
    if 'dones' in localvars:
        for i in range(len(localvars['current_lengths'])):
            if localvars['current_lengths'][i] >= 1000:
                localvars['dones'][i] = True
    return

# Function to evaluate the advantage of the policy
def advantage_evaluation(model, args, horizon=1000):
    if FQE is None:
        raise ImportError(
            "advantage_evaluation requires stable_baselines3.common.fqe.FQE from the "
            "custom SB3/FQE build used by the original project."
        )
    fqe = FQE(model.replay_buffer.obs_shape[0], model.replay_buffer.action_dim, lr=1e-4, gamma=model.gamma, device=str(device))
    fqe.build_q_net(model)
    q_loss = fqe.train(fqe.model, 256, 10)
    s0 = [model.env.reset(seed=args.seed) for _ in range(100)]
    s0 = FloatTensor(s0)
    s0 = s0.squeeze(1)
    act, _, _ = model.policy(s0)
    q_pred = fqe.predict(s0, act.detach(), horizon=horizon)
    # print(f'q_pred: {q_pred}, q_loss: {q_loss}')
    return q_pred, q_loss

# PPO Warppaer to wrap our empty space agent for calculating FQE
# NOTE ON EXACT TARGET-ACTION CACHING:
# pi(s') is fixed for each candidate and is mathematically cacheable. However,
# d3rlpy's public FQE path calls predict_best_action(next_observations) without
# exposing the sampled dataset indices. Exact indexed caching would require
# modifying/subclassing d3rlpy FQE internals. To preserve the estimator itself,
# this optimized version uses direct batched GPU inference instead of a brittle
# observation-hash cache or a custom replacement FQE implementation.
class PPOQWrapper(QLearningAlgoBase):
    def __init__(self, ppo_policy):
        super().__init__(config=LearnableConfig(), device=str(ppo_policy.device), enable_ddp=False)
        self.ppo = ppo_policy
        self._action_space = self.get_action_type()  # Explicitly set action space
        self.device = str(ppo_policy.device) # Store device for consistency
        print(f"PPOQWrapper initialized on device: {self.device}")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS # Default or raise error

    # --- Critical Overrides ---
    @torch.no_grad()
    def predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        """Evaluate the fixed PPO target policy directly in PyTorch.

        This removes the previous CUDA -> CPU/NumPy -> SB3.predict -> CUDA
        round trip from every FQE Bellman update while keeping the same
        deterministic PPO policy and Box-action post-processing.
        """
        # Continuous-control experiments in this file use tensor observations.
        # Keep a compatibility fallback for structured observations.
        if isinstance(x, (list, tuple)):
            x_np = [
                xi.detach().cpu().numpy() if torch.is_tensor(xi) else xi
                for xi in x
            ]
            actions, _ = self.ppo.predict(x_np, deterministic=True)
            return torch.as_tensor(
                actions, dtype=torch.float32, device=self.ppo.device
            )

        x = x.to(device=self.ppo.device, dtype=torch.float32)

        # SB3 BasePolicy.predict switches the policy to evaluation mode first.
        # Preserve that behavior for exact deterministic inference.
        self.ppo.policy.set_training_mode(False)
        actions = self.ppo.policy._predict(x, deterministic=True)

        action_space = self.ppo.action_space
        if isinstance(action_space, gym.spaces.Box):
            low = torch.as_tensor(
                action_space.low, dtype=actions.dtype, device=actions.device
            )
            high = torch.as_tensor(
                action_space.high, dtype=actions.dtype, device=actions.device
            )

            if self.ppo.policy.squash_output:
                # Same mapping as SB3 BasePolicy.unscale_action, kept on GPU.
                actions = low + 0.5 * (actions + 1.0) * (high - low)
            else:
                # Same clipping performed by SB3 BasePolicy.predict.
                actions = torch.maximum(torch.minimum(actions, high), low)

        return actions

    @torch.no_grad()
    def predict_value(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        """Directly access PPO's critic network."""
        # Use the critic network as in your original code
        if hasattr(self.ppo.policy, 'value_net'):
            return self.ppo.policy.value_net(x)
        else:
            raise AttributeError("PPO critic network not found")

    # --- Required Base Class Methods ---
    def inner_create_impl(self, observation_shape, action_size):
        # Directly use PPO networks instead of dummy
        self._impl = self  # Bypass d3rlpy's impl requirement

    def update(self, batch: TorchMiniBatch) -> dict:
        """No-op since PPO isn't being trained."""
        return {}



class BatchedFQECritic(nn.Module):
    """Vectorized bank of candidate-by-ensemble continuous-action FQE critics.

    The architecture mirrors d3rlpy 2.8.1's default vector continuous
    MeanQFunction:
        concat(obs, action) -> 256 ReLU -> 256 ReLU -> 1

    For LCB-FQE each candidate owns ``n_ensemble`` independent critics. The
    ensemble member initializations differ from one another, but the same set
    of member initializations is reused for every candidate. This common-random-
    numbers design prevents candidate ranking from being confounded by giving
    different candidates different initialization luck.

    When n_ensemble == 1, member 0 follows the exact initialization sequence
    used by the previous BatchedFQECritic implementation.
    """

    def __init__(
        self,
        n_candidates,
        observation_dim,
        action_dim,
        hidden_units=(256, 256),
        n_ensemble=1,
        compute_device=None,
    ):
        super().__init__()
        if len(hidden_units) != 2:
            raise ValueError(
                "Native batched FQE currently expects exactly two hidden layers."
            )

        self.n_candidates = int(n_candidates)
        self.n_ensemble = int(n_ensemble)
        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be > 0.")
        if self.n_ensemble <= 0:
            raise ValueError("n_ensemble must be > 0.")

        self.n_critics = self.n_candidates * self.n_ensemble
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_units = tuple(int(v) for v in hidden_units)
        self.compute_device = (
            compute_device if compute_device is not None else device
        )

        h1, h2 = self.hidden_units
        input_dim = self.observation_dim + self.action_dim

        # Build B independent templates on CPU using torch.nn.Linear's default
        # initialization. The first member intentionally consumes RNG in exactly
        # the same order as the old single-critic implementation:
        # fc1 -> fc2 -> dummy encoder-size draw -> output layer.
        ensemble_templates = []
        for _ in range(self.n_ensemble):
            template_fc1 = nn.Linear(input_dim, h1)
            template_fc2 = nn.Linear(h1, h2)

            with torch.no_grad():
                dummy_obs = torch.rand(2, self.observation_dim)
                dummy_action = torch.rand(2, self.action_dim)
                dummy = torch.cat((dummy_obs, dummy_action), dim=-1)
                dummy = torch.relu(template_fc1(dummy))
                dummy = torch.relu(template_fc2(dummy))
                del dummy

            template_out = nn.Linear(h2, 1)
            ensemble_templates.append(
                (
                    template_fc1.weight.detach().clone(),
                    template_fc1.bias.detach().clone(),
                    template_fc2.weight.detach().clone(),
                    template_fc2.bias.detach().clone(),
                    template_out.weight.detach().clone(),
                    template_out.bias.detach().clone(),
                )
            )

        def candidate_repeated_parameter(template_index):
            # First stack distinct ensemble members:
            #   [ensemble, ...]
            # then reuse that same ordered ensemble for every candidate:
            #   [candidate, ensemble, ...] -> [candidate * ensemble, ...].
            per_ensemble = torch.stack(
                [templates[template_index] for templates in ensemble_templates],
                dim=0,
            )
            expanded = (
                per_ensemble.unsqueeze(0)
                .repeat(
                    self.n_candidates,
                    1,
                    *([1] * (per_ensemble.ndim - 1)),
                )
                .reshape(self.n_critics, *per_ensemble.shape[1:])
                .to(self.compute_device)
                .clone()
            )
            return nn.Parameter(expanded)

        # Flatten candidate x ensemble into one critic-bank dimension for the
        # batched matrix multiplies.
        self.w1 = candidate_repeated_parameter(0)
        self.b1 = candidate_repeated_parameter(1)
        self.w2 = candidate_repeated_parameter(2)
        self.b2 = candidate_repeated_parameter(3)
        self.w3 = candidate_repeated_parameter(4)
        self.b3 = candidate_repeated_parameter(5)

    def _expand_to_critic_bank(self, tensor, feature_name):
        """Map shared/candidate inputs to [candidate*ensemble, batch, dim]."""
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).expand(self.n_critics, -1, -1)

        if tensor.ndim != 3:
            raise ValueError(
                f"{feature_name} must have rank 2 or 3, got shape "
                f"{tuple(tensor.shape)}."
            )

        if tensor.shape[0] == self.n_critics:
            return tensor

        if tensor.shape[0] == self.n_candidates:
            return (
                tensor.unsqueeze(1)
                .expand(-1, self.n_ensemble, -1, -1)
                .reshape(self.n_critics, tensor.shape[1], tensor.shape[2])
            )

        raise ValueError(
            f"{feature_name} leading dimension must be n_candidates="
            f"{self.n_candidates} or n_critics={self.n_critics}; got "
            f"{tensor.shape[0]}."
        )

    def forward(self, observations, actions):
        """Return Q values shaped [candidate, ensemble, batch, 1].

        observations can be:
          [batch, obs_dim]                         shared replay observations
          [candidate, batch, obs_dim]             candidate-specific observations
          [candidate*ensemble, batch, obs_dim]    fully expanded observations

        actions can be:
          [batch, action_dim]                      shared replay actions
          [candidate, batch, action_dim]          candidate-policy actions
          [candidate*ensemble, batch, action_dim] fully expanded actions
        """
        observations = self._expand_to_critic_bank(
            observations, "observations"
        )
        actions = self._expand_to_critic_bank(actions, "actions")

        x = torch.cat((observations, actions), dim=-1)
        x = torch.bmm(x, self.w1.transpose(1, 2)) + self.b1.unsqueeze(1)
        x = torch.relu(x)
        x = torch.bmm(x, self.w2.transpose(1, 2)) + self.b2.unsqueeze(1)
        x = torch.relu(x)
        q = torch.bmm(x, self.w3.transpose(1, 2)) + self.b3.unsqueeze(1)

        return q.reshape(
            self.n_candidates,
            self.n_ensemble,
            q.shape[1],
            q.shape[2],
        )

def _policy_actions_current_model(model, observations, chunk_size=None):
    """Compute deterministic SB3 PPO actions entirely in PyTorch.

    This reproduces BasePolicy.predict's deterministic Box-action postprocessing
    while avoiding CPU/NumPy transfers.
    """
    if chunk_size is None:
        chunk_size = NATIVE_FQE_ACTION_CHUNK_SIZE

    if not torch.is_tensor(observations):
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=device
        )
    else:
        observations = observations.to(device=device, dtype=torch.float32)

    model.policy.set_training_mode(False)
    action_space = model.action_space

    outputs = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], chunk_size):
            obs_batch = observations[start : start + chunk_size]
            actions = model.policy._predict(obs_batch, deterministic=True)

            if isinstance(action_space, gym.spaces.Box):
                low = torch.as_tensor(
                    action_space.low,
                    dtype=actions.dtype,
                    device=actions.device,
                )
                high = torch.as_tensor(
                    action_space.high,
                    dtype=actions.dtype,
                    device=actions.device,
                )

                if model.policy.squash_output:
                    actions = low + 0.5 * (actions + 1.0) * (high - low)
                else:
                    actions = torch.maximum(torch.minimum(actions, high), low)

            outputs.append(actions)

    return torch.cat(outputs, dim=0)



def build_native_fqe_replay_data(
    model,
    max_transitions=None,
    finite_horizon_steps=None,
):
    """Freeze scientifically correct replay transitions for native batched FQE.

    Unlike d3rlpy's MDPDataset/Episode representation, this path can retain the
    FINAL transition of TimeLimit-truncated episodes because ReplayBuffer stores
    an explicit next_observation for every transition.

    This function therefore uses the corrected replay semantics directly:
      * executed environment action
      * raw environment reward
      * true terminal/truncated next observation
      * original true-terminal and timeout flags
      * one-step transition interval
      * per-episode timestep t reconstructed from real replay boundaries

    In time-conditioned finite-horizon mode, ``finite_horizon_steps`` is the
    online evaluation horizon H. The native FQE critic receives t/H and treats
    BOTH true terminals and truncations as zero-bootstrap boundaries. Replay
    storage itself is unchanged.

    If max_transitions is not None, select the most recent portion of the
    chronological replay ring first, then discard leading/trailing partial
    episodes. Therefore the returned number of transitions can be smaller than
    the requested limit. max_transitions=None preserves full-buffer behavior.
    """
    rb = model.replay_buffer

    if TIME_CONDITIONED_FINITE_HORIZON_FQE:
        if finite_horizon_steps is None:
            raise ValueError(
                "finite_horizon_steps is required when "
                "TIME_CONDITIONED_FINITE_HORIZON_FQE=1."
            )
        finite_horizon_steps = int(finite_horizon_steps)
        if finite_horizon_steps <= 0:
            raise ValueError("finite_horizon_steps must be > 0.")
    elif finite_horizon_steps is not None:
        finite_horizon_steps = int(finite_horizon_steps)

    semantics_version = getattr(
        rb, "_fqe_replay_semantics_version", None
    )
    if semantics_version != FQE_REPLAY_SEMANTICS_VERSION:
        raise RuntimeError(
            "Native FQE requires replay semantics version "
            f"{FQE_REPLAY_SEMANTICS_VERSION}, got {semantics_version}. "
            "Install the replay-semantics patch and regenerate/load a corrected "
            "initial replay buffer."
        )

    if rb.size() == 0:
        raise RuntimeError(
            "Cannot build native FQE data from an empty replay buffer."
        )

    if not hasattr(rb, "next_observations"):
        raise RuntimeError(
            "Native FQE replay-data path requires explicit "
            "ReplayBuffer.next_observations."
        )

    if rb.full:
        time_indices = np.concatenate(
            (
                np.arange(rb.pos, rb.buffer_size, dtype=np.int64),
                np.arange(0, rb.pos, dtype=np.int64),
            )
        )
    else:
        time_indices = np.arange(0, rb.pos, dtype=np.int64)

    total_available_transitions = int(len(time_indices) * rb.n_envs)
    window_was_truncated = False
    leading_partial_by_env = None

    if max_transitions is not None:
        max_transitions = int(max_transitions)
        if max_transitions <= 0:
            raise ValueError("max_transitions must be > 0 or None.")

        # One replay position contains one transition per vectorized environment.
        # Keep only the newest complete set of VecEnv positions and never exceed
        # the requested total transition budget.
        positions_to_keep = max_transitions // rb.n_envs
        if positions_to_keep <= 0:
            raise ValueError(
                f"Replay window {max_transitions} is smaller than n_envs={rb.n_envs}."
            )
        if positions_to_keep < len(time_indices):
            # Because this finite window is cut from an already reconstructed
            # chronological history, we can inspect its immediate predecessor.
            # If that predecessor is an episode boundary, the first selected
            # transition is a genuine episode start and should NOT be discarded.
            predecessor_index = time_indices[-positions_to_keep - 1]
            predecessor_dones = np.asarray(rb.dones)[predecessor_index]
            predecessor_timeouts = np.asarray(rb.timeouts)[predecessor_index]
            predecessor_boundaries = (
                (np.asarray(predecessor_dones).reshape(-1) > 0.5)
                | (np.asarray(predecessor_timeouts).reshape(-1) > 0.5)
            )
            leading_partial_by_env = ~predecessor_boundaries

            time_indices = time_indices[-positions_to_keep:]
            window_was_truncated = True

    selected_raw_transitions = int(len(time_indices) * rb.n_envs)

    # Detect the known discontinuity created by replay-buffer reload followed
    # by an environment reset. This is NOT an artificial MDP terminal. It is a
    # data-continuity marker telling FQE that the saved unfinished trajectory
    # cannot be joined to the first trajectory collected after the reset.
    resume_seam_local_index = None
    resume_seam_pos = getattr(rb, "_fqe_resume_seam_pos", None)
    resume_positions_written = int(
        getattr(rb, "_fqe_resume_positions_written", 0)
    )
    if (
        TIME_CONDITIONED_FINITE_HORIZON_FQE
        and resume_seam_pos is not None
        and resume_positions_written > 0
        and resume_positions_written < rb.buffer_size
    ):
        seam_matches = np.flatnonzero(
            time_indices == int(resume_seam_pos)
        )
        if len(seam_matches) > 1:
            raise RuntimeError(
                "Internal FQE replay error: resume seam appeared more than "
                "once in chronological replay indices."
            )
        if len(seam_matches) == 1:
            resume_seam_local_index = int(seam_matches[0])

    obs_raw = np.asarray(rb.observations)[time_indices]
    next_obs_raw = np.asarray(rb.next_observations)[time_indices]
    actions_raw = np.asarray(rb.actions)[time_indices]
    rewards_raw = np.asarray(rb.rewards)[time_indices]
    dones_raw = np.asarray(rb.dones)[time_indices]
    timeouts_raw = np.asarray(rb.timeouts)[time_indices]

    observations_parts = []
    next_observations_parts = []
    actions_parts = []
    rewards_parts = []
    terminals_parts = []
    timeouts_parts = []
    timestep_parts = []
    next_timestep_parts = []
    initial_observation_parts = []
    initial_timestep_parts = []
    resume_gap_discarded_transitions = 0

    for env_idx in range(rb.n_envs):
        obs_env = np.asarray(obs_raw[:, env_idx]).copy()
        next_obs_env = np.asarray(next_obs_raw[:, env_idx]).copy()
        actions_env = np.asarray(actions_raw[:, env_idx]).copy()
        rewards_env = np.asarray(
            rewards_raw[:, env_idx]
        ).reshape(-1, 1).astype(np.float32, copy=True)
        dones_env = np.asarray(
            dones_raw[:, env_idx]
        ).reshape(-1, 1).astype(np.float32, copy=True)
        timeouts_env = np.asarray(
            timeouts_raw[:, env_idx]
        ).reshape(-1, 1).astype(np.float32, copy=True)

        terminals_env = dones_env * (1.0 - timeouts_env)

        # If this selected replay view spans the load/reset seam, remove only
        # the unfinished OLD trajectory tail immediately before the seam. The
        # first transition at the seam was collected after an explicit env
        # reset, so it is a known episode start. Complete saved episodes before
        # that tail are retained, and all post-reset data are retained.
        #
        # Example of the invalid raw chronology:
        #   ... [old complete boundary] old_partial_tail |RESET| new_ep ...
        # We convert it to:
        #   ... [old complete boundary] | new_ep ...
        # without inventing a terminal reward/transition.
        begins_at_known_resume_start = False
        if resume_seam_local_index is not None:
            seam = int(resume_seam_local_index)
            if not (0 <= seam < len(obs_env)):
                raise RuntimeError(
                    "Internal FQE replay error: resume seam index is outside "
                    "the selected replay view."
                )

            raw_boundary_flags = (
                (terminals_env[:, 0] > 0.5)
                | (timeouts_env[:, 0] > 0.5)
            )
            boundaries_before_seam = np.flatnonzero(
                raw_boundary_flags[:seam]
            )
            prefix_end = (
                int(boundaries_before_seam[-1] + 1)
                if len(boundaries_before_seam) > 0
                else 0
            )

            discarded_here = int(seam - prefix_end)
            if discarded_here < 0:
                raise RuntimeError(
                    "Internal FQE replay error: negative resume-gap length."
                )

            if discarded_here > 0 or seam == 0:
                def _join_across_resume(arr):
                    return np.concatenate(
                        (arr[:prefix_end], arr[seam:]), axis=0
                    )

                obs_env = _join_across_resume(obs_env)
                next_obs_env = _join_across_resume(next_obs_env)
                actions_env = _join_across_resume(actions_env)
                rewards_env = _join_across_resume(rewards_env)
                dones_env = _join_across_resume(dones_env)
                timeouts_env = _join_across_resume(timeouts_env)
                terminals_env = _join_across_resume(terminals_env)

                resume_gap_discarded_transitions += discarded_here
                begins_at_known_resume_start = (prefix_end == 0)

        boundary_flags = (
            (terminals_env[:, 0] > 0.5)
            | (timeouts_env[:, 0] > 0.5)
        )
        boundaries = np.flatnonzero(boundary_flags)

        if len(boundaries) == 0:
            raise RuntimeError(
                f"No complete episode boundary was found in replay-buffer "
                f"env {env_idx}."
            )

        # Same complete-episode trimming policy as build_fqe_dataset().
        # For the full circular replay we conservatively discard the leading
        # segment because its predecessor has been overwritten. For a finite
        # recent window, the predecessor is still available and tells us
        # exactly whether the first selected transition begins a new episode.
        if begins_at_known_resume_start:
            # The environment was explicitly reset before this transition.
            leading_partial = False
        elif window_was_truncated:
            leading_partial = bool(leading_partial_by_env[env_idx])
        else:
            leading_partial = bool(rb.full)

        start = int(boundaries[0] + 1) if leading_partial else 0
        end = int(boundaries[-1] + 1)

        if start >= end:
            raise RuntimeError(
                f"Replay-buffer env {env_idx} contains no complete episode "
                "after removing partial circular-buffer trajectories."
            )

        obs_keep = obs_env[start:end]
        next_obs_keep = next_obs_env[start:end]
        actions_keep = actions_env[start:end]
        rewards_keep = rewards_env[start:end]
        terminals_keep = terminals_env[start:end]
        timeouts_keep = timeouts_env[start:end]

        kept_boundary_flags = (
            (terminals_keep[:, 0] > 0.5)
            | (timeouts_keep[:, 0] > 0.5)
        )
        kept_boundaries = np.flatnonzero(kept_boundary_flags)

        # Every retained segment is a concatenation of complete episodes.
        # Episode starts are transition 0 and immediately after each boundary
        # except the final boundary.
        episode_starts = [0]
        episode_starts.extend(
            int(idx + 1)
            for idx in kept_boundaries[:-1]
        )

        # Reconstruct the within-episode timestep without changing replay
        # collection. Each complete episode contributes t=0,...,L-1 and the
        # corresponding next-timestep t+1. This is used only by the FQE critic.
        timesteps_keep = np.empty(
            (len(obs_keep), 1), dtype=np.float32
        )
        next_timesteps_keep = np.empty(
            (len(obs_keep), 1), dtype=np.float32
        )
        episode_start = 0
        for boundary_index in kept_boundaries:
            episode_end = int(boundary_index + 1)
            episode_length = int(episode_end - episode_start)
            if episode_length <= 0:
                raise RuntimeError(
                    "Encountered an empty episode while reconstructing FQE time."
                )
            if (
                TIME_CONDITIONED_FINITE_HORIZON_FQE
                and episode_length > finite_horizon_steps
            ):
                raise RuntimeError(
                    "Replay episode is longer than the configured finite "
                    f"horizon: length={episode_length}, "
                    f"H={finite_horizon_steps}."
                )

            episode_t = np.arange(
                episode_length, dtype=np.float32
            ).reshape(-1, 1)
            timesteps_keep[episode_start:episode_end] = episode_t
            next_timesteps_keep[episode_start:episode_end] = episode_t + 1.0
            episode_start = episode_end

        if episode_start != len(obs_keep):
            raise RuntimeError(
                "Internal FQE timestep reconstruction error: retained replay "
                "segment did not end on an episode boundary."
            )

        initial_indices = np.asarray(episode_starts, dtype=np.int64)
        initial_observation_parts.append(obs_keep[initial_indices])
        initial_timestep_parts.append(
            np.zeros((len(initial_indices), 1), dtype=np.float32)
        )

        observations_parts.append(obs_keep)
        next_observations_parts.append(next_obs_keep)
        actions_parts.append(actions_keep)
        rewards_parts.append(rewards_keep)
        terminals_parts.append(terminals_keep)
        timeouts_parts.append(timeouts_keep)
        timestep_parts.append(timesteps_keep)
        next_timestep_parts.append(next_timesteps_keep)

    observations = np.concatenate(
        observations_parts, axis=0
    ).astype(np.float32, copy=False)
    next_observations = np.concatenate(
        next_observations_parts, axis=0
    ).astype(np.float32, copy=False)
    actions = np.concatenate(
        actions_parts, axis=0
    ).astype(np.float32, copy=False)
    rewards = np.concatenate(
        rewards_parts, axis=0
    ).astype(np.float32, copy=False)
    terminals = np.concatenate(
        terminals_parts, axis=0
    ).astype(np.float32, copy=False)
    timeouts = np.concatenate(
        timeouts_parts, axis=0
    ).astype(np.float32, copy=False)
    timesteps = np.concatenate(
        timestep_parts, axis=0
    ).astype(np.float32, copy=False)
    next_timesteps = np.concatenate(
        next_timestep_parts, axis=0
    ).astype(np.float32, copy=False)
    initial_observations = np.concatenate(
        initial_observation_parts, axis=0
    ).astype(np.float32, copy=False)
    initial_timesteps = np.concatenate(
        initial_timestep_parts, axis=0
    ).astype(np.float32, copy=False)

    if np.any(
        np.logical_and(
            terminals[:, 0] > 0.5,
            timeouts[:, 0] > 0.5,
        )
    ):
        raise RuntimeError(
            "Internal native FQE data error: a transition is marked as both "
            "terminal and timeout."
        )

    # Every stored transition is one environment step in this PPO replay buffer.
    intervals = np.ones_like(
        rewards, dtype=np.float32
    )

    n_episodes = int(
        np.sum(
            (terminals[:, 0] > 0.5)
            | (timeouts[:, 0] > 0.5)
        )
    )

    coverage_label = "full" if max_transitions is None else str(int(max_transitions))

    if TIME_CONDITIONED_FINITE_HORIZON_FQE:
        if np.any(timesteps[:, 0] < 0.0):
            raise RuntimeError("Negative reconstructed FQE timestep detected.")
        if np.any(timesteps[:, 0] >= float(finite_horizon_steps)):
            raise RuntimeError(
                "Reconstructed FQE timestep reaches/exceeds the configured "
                f"horizon H={finite_horizon_steps}."
            )
        if np.any(next_timesteps[:, 0] > float(finite_horizon_steps)):
            raise RuntimeError(
                "Reconstructed next FQE timestep exceeds the configured "
                f"horizon H={finite_horizon_steps}."
            )

    print(
        "Native FQE replay data "
        f"[coverage={coverage_label}]: "
        f"{len(rewards)} complete-episode transitions, "
        f"{int(np.sum(terminals))} true terminals, "
        f"{int(np.sum(timeouts))} timeouts, "
        f"{n_episodes} episodes, "
        f"{len(initial_observations)} initial states"
    )
    if resume_gap_discarded_transitions > 0:
        print(
            "  FQE replay-resume seam: discarded "
            f"{resume_gap_discarded_transitions} transition(s) from the "
            "unfinished saved trajectory immediately before the post-load "
            "environment reset."
        )
    if TIME_CONDITIONED_FINITE_HORIZON_FQE:
        print(
            "  finite-horizon FQE time reconstruction: "
            f"H={finite_horizon_steps}, "
            f"t_range=[{int(np.min(timesteps))}, {int(np.max(timesteps))}], "
            "critic_time_feature=t/H"
        )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
        "timeouts": timeouts,
        "intervals": intervals,
        "timesteps": timesteps,
        "next_timesteps": next_timesteps,
        "initial_observations": initial_observations,
        "initial_timesteps": initial_timesteps,
        "finite_horizon_steps": (
            int(finite_horizon_steps)
            if finite_horizon_steps is not None
            else None
        ),
        "time_conditioned_finite_horizon": bool(
            TIME_CONDITIONED_FINITE_HORIZON_FQE
        ),
        "coverage_label": coverage_label,
        "requested_max_transitions": max_transitions,
        "selected_raw_transitions": selected_raw_transitions,
        "actual_transitions": int(len(rewards)),
        "n_episodes": n_episodes,
        "total_available_transitions": total_available_transitions,
        "resume_gap_discarded_transitions": int(
            resume_gap_discarded_transitions
        ),
    }


def build_native_fqe_data(dataset):
    """Materialize exactly the transitions exposed by the d3rlpy MDPDataset.

    This intentionally goes through dataset.buffer + dataset.transition_picker
    instead of reconstructing a second interpretation of episode boundaries.
    Therefore the native FQE sees the same transition population that d3rlpy's
    sample_transition_batch() sees.

    It also reproduces InitialStateValueEstimationEvaluator's initial-state
    selection: the first transition of every 1024-transition evaluation window.
    Ant episodes are normally shorter than 1024, so this is one initial state
    per complete episode.
    """
    transition_count = int(dataset.transition_count)
    if transition_count <= 0:
        raise RuntimeError("Native FQE received an empty d3rlpy dataset.")

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    intervals = []

    # ReplayBuffer.sample_transition() selects an integer in
    # [0, transition_count) and looks up dataset.buffer[index]. Keeping this
    # exact ordering allows one shared integer-index minibatch to reproduce
    # d3rlpy's transition population for all candidate critics.
    for index in range(transition_count):
        episode, transition_index = dataset.buffer[index]
        transition = dataset.transition_picker(episode, transition_index)

        if not isinstance(transition.observation, np.ndarray):
            raise NotImplementedError(
                "Native batched FQE currently supports flat NumPy observations "
                "(the active Ant-v5 setup)."
            )
        if not isinstance(transition.next_observation, np.ndarray):
            raise NotImplementedError(
                "Native batched FQE currently supports flat NumPy observations."
            )

        observations.append(np.asarray(transition.observation))
        actions.append(np.asarray(transition.action))
        rewards.append(np.asarray(transition.reward).reshape(-1))
        next_observations.append(np.asarray(transition.next_observation))
        terminals.append(float(transition.terminal))
        intervals.append(int(transition.interval))

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    next_observations = np.asarray(next_observations, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32).reshape(-1, 1)
    intervals = np.asarray(intervals, dtype=np.float32).reshape(-1, 1)

    # Mirror d3rlpy.metrics.InitialStateValueEstimationEvaluator exactly.
    initial_observations = []
    evaluator_window_size = 1024
    for episode in dataset.episodes:
        # d3rlpy make_batches() computes the number of windows from len(episode)
        # and clips each window by episode.transition_count.
        n_batches = len(episode) // evaluator_window_size
        if len(episode) % evaluator_window_size != 0:
            n_batches += 1

        for batch_index in range(n_batches):
            head_index = batch_index * evaluator_window_size
            last_index = min(
                head_index + evaluator_window_size,
                int(episode.transition_count),
            )
            # A valid d3rlpy evaluation batch must contain at least one transition.
            if head_index >= last_index:
                continue

            transition = dataset.transition_picker(episode, head_index)
            if not isinstance(transition.observation, np.ndarray):
                raise NotImplementedError(
                    "Native batched FQE currently supports flat NumPy observations."
                )
            initial_observations.append(
                np.asarray(transition.observation, dtype=np.float32)
            )

    if not initial_observations:
        raise RuntimeError("No initial states were found for native FQE evaluation.")

    initial_observations = np.asarray(initial_observations, dtype=np.float32)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
        "intervals": intervals,
        "initial_observations": initial_observations,
    }


class LCBFQEScores(list):
    """List-compatible LCB scores with analysis-only uncertainty diagnostics."""

    def __init__(
        self,
        scores,
        mean_q,
        mean_sigma,
        ensemble_member_values,
        beta,
        ensemble_size,
    ):
        super().__init__(float(v) for v in scores)
        self.mean_q = np.asarray(mean_q, dtype=np.float64)
        self.mean_sigma = np.asarray(mean_sigma, dtype=np.float64)
        self.ensemble_member_values = np.asarray(
            ensemble_member_values, dtype=np.float64
        )
        self.beta = float(beta)
        self.ensemble_size = int(ensemble_size)


class SupportPenalizedFQEScores(list):
    """List-compatible support-penalized FQE scores with diagnostics.

    ``mean_q`` is the ordinary FQE value estimate used as the base score. For
    B>1 it is the ensemble-mean FQE value; LCB is intentionally not used for
    ranking in this experiment.
    """

    def __init__(
        self,
        scores,
        mean_q,
        action_divergence,
        support_penalty,
        penalty_lambda,
        support_reference_label,
        support_reference_transitions,
        mean_sigma=None,
        ensemble_member_values=None,
        ensemble_size=1,
    ):
        super().__init__(float(v) for v in scores)
        self.mean_q = np.asarray(mean_q, dtype=np.float64)
        self.action_divergence = np.asarray(
            action_divergence, dtype=np.float64
        )
        self.support_penalty = np.asarray(
            support_penalty, dtype=np.float64
        )
        self.penalty_lambda = float(penalty_lambda)
        self.support_reference_label = str(support_reference_label)
        self.support_reference_transitions = int(
            support_reference_transitions
        )
        if mean_sigma is None:
            mean_sigma = np.zeros_like(self.mean_q)
        self.mean_sigma = np.asarray(mean_sigma, dtype=np.float64)
        if ensemble_member_values is None:
            ensemble_member_values = self.mean_q.reshape(-1, 1)
        self.ensemble_member_values = np.asarray(
            ensemble_member_values, dtype=np.float64
        )
        self.ensemble_size = int(ensemble_size)
        # Compatibility attributes for older downstream diagnostics.
        self.beta = 0.0


def compute_behavior_action_divergence(model, agents, support_data):
    """Compute E_D[||pi(s)-a_buffer||_2^2] for every candidate policy.

    The support reference is frozen replay data built before online candidate
    evaluation. ``support_data['actions']`` therefore contains the corrected
    action actually executed in the environment, not PPO's pre-clipped action.

    This function is analysis-only. It restores the model policy weights after
    evaluating all candidates so it cannot affect PPO/ESA/online selection.
    """
    if len(agents) == 0:
        return np.empty(0, dtype=np.float64)

    support_obs_cpu = np.asarray(support_data["observations"], dtype=np.float32)
    support_actions_cpu = np.asarray(support_data["actions"], dtype=np.float32)

    if support_obs_cpu.shape[0] != support_actions_cpu.shape[0]:
        raise RuntimeError(
            "Support-reference observation/action length mismatch: "
            f"{support_obs_cpu.shape[0]} vs {support_actions_cpu.shape[0]}."
        )
    if support_obs_cpu.shape[0] == 0:
        raise RuntimeError("Support-reference replay data is empty.")

    support_observations = torch.as_tensor(
        support_obs_cpu, dtype=torch.float32, device=device
    )
    support_actions = torch.as_tensor(
        support_actions_cpu, dtype=torch.float32, device=device
    )

    original_policy_state = state_dict_to_cpu(model.policy.state_dict())
    divergences = []

    try:
        for candidate_index, agent in enumerate(agents):
            model.policy.load_state_dict(agent)
            model.policy.to(device)

            candidate_actions = _policy_actions_current_model(
                model, support_observations
            )
            if candidate_actions.shape != support_actions.shape:
                raise RuntimeError(
                    "Candidate/buffer action shape mismatch for support penalty "
                    f"at candidate {candidate_index}: "
                    f"candidate={tuple(candidate_actions.shape)}, "
                    f"buffer={tuple(support_actions.shape)}."
                )

            # Literal squared L2 norm: sum across action dimensions, then
            # expectation across frozen replay transitions.
            squared_l2 = (candidate_actions - support_actions).pow(2).sum(dim=-1)
            divergence = squared_l2.mean()

            if not torch.isfinite(divergence):
                raise RuntimeError(
                    "Support penalty produced a non-finite action divergence "
                    f"for candidate {candidate_index}."
                )

            divergences.append(float(divergence.detach().cpu().item()))
    finally:
        model.policy.load_state_dict(original_policy_state)
        model.policy.to(device)

    return np.asarray(divergences, dtype=np.float64)


def compute_behavior_action_divergence_preserving_rng(
    model, agents, support_data
):
    """Compute support divergence without perturbing experiment RNG state."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    try:
        return compute_behavior_action_divergence(
            model, agents, support_data
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)



def _deterministic_even_subsample_indices(n_items, max_items):
    """Even deterministic subsample with no experiment-RNG consumption."""
    n_items = int(n_items)
    max_items = int(max_items)
    if n_items <= 0 or max_items <= 0:
        raise ValueError("Subsample sizes must be positive.")
    if max_items >= n_items:
        return np.arange(n_items, dtype=np.int64)

    # floor(j * n / m), j=0,...,m-1 is unique whenever m <= n and spans the
    # full chronological replay view without consuming NumPy RNG state.
    return (
        (np.arange(max_items, dtype=np.int64) * n_items) // max_items
    ).astype(np.int64, copy=False)


def compute_state_conditional_knn_support(model, agents, support_data):
    """Estimate candidate action support conditioned on nearby replay states.

    Replay observations are standardized feature-wise before Euclidean kNN so
    high-scale observation coordinates do not dominate the state metric. Query
    states are an evenly spaced deterministic subset of the full frozen replay
    view. The query transition itself is explicitly excluded from its neighbor
    set, making the behavior calibration leave-one-out rather than collapsing
    back to the old paired-action metric.

    The returned candidate metric is the nearest local replay-action squared L2
    distance at each query state. The behavior threshold is calibrated from the
    same local neighborhoods using the replay action at the query state.

    Analysis only: candidate policy weights are restored before returning.
    """
    if len(agents) == 0:
        raise RuntimeError("kNN support study received zero candidate policies.")
    if not isinstance(model.action_space, gym.spaces.Box):
        raise NotImplementedError(
            "State-conditional kNN support currently supports Box actions only."
        )

    observations_cpu = np.asarray(
        support_data["observations"], dtype=np.float32
    )
    actions_cpu = np.asarray(support_data["actions"], dtype=np.float32)

    if observations_cpu.ndim != 2 or actions_cpu.ndim != 2:
        raise NotImplementedError(
            "State-conditional kNN support currently expects flat vector "
            "observations and actions."
        )
    if observations_cpu.shape[0] != actions_cpu.shape[0]:
        raise RuntimeError(
            "kNN support observation/action length mismatch: "
            f"{observations_cpu.shape[0]} vs {actions_cpu.shape[0]}."
        )
    if not np.all(np.isfinite(observations_cpu)):
        raise RuntimeError("kNN support replay observations contain non-finite values.")
    if not np.all(np.isfinite(actions_cpu)):
        raise RuntimeError("kNN support replay actions contain non-finite values.")

    n_reference = int(observations_cpu.shape[0])
    if n_reference <= KNN_SUPPORT_K:
        raise RuntimeError(
            "kNN support requires more replay transitions than KNN_SUPPORT_K: "
            f"reference={n_reference}, k={KNN_SUPPORT_K}."
        )

    query_indices = _deterministic_even_subsample_indices(
        n_reference,
        min(KNN_SUPPORT_QUERY_COUNT, n_reference),
    )
    n_query = int(len(query_indices))

    # State standardization is fitted only on the frozen replay reference.
    state_mean = observations_cpu.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = observations_cpu.std(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.where(state_std > 1e-6, state_std, 1.0).astype(np.float32)
    standardized_obs_cpu = (
        (observations_cpu - state_mean) / state_std
    ).astype(np.float32, copy=False)

    reference_states = torch.as_tensor(
        standardized_obs_cpu, dtype=torch.float32, device=device
    )
    reference_actions = torch.as_tensor(
        actions_cpu, dtype=torch.float32, device=device
    )
    query_index_tensor = torch.as_tensor(
        query_indices, dtype=torch.long, device=device
    )
    query_states = reference_states[query_index_tensor]
    query_behavior_actions = reference_actions[query_index_tensor]

    # Exact GPU/CPU torch kNN in bounded query chunks. The self transition is
    # assigned +inf before top-k, so every neighbor is a DIFFERENT replay row.
    neighbor_index_chunks = []
    neighbor_distance_chunks = []
    with torch.no_grad():
        for start in range(0, n_query, KNN_SUPPORT_QUERY_CHUNK_SIZE):
            end = min(start + KNN_SUPPORT_QUERY_CHUNK_SIZE, n_query)
            query_chunk = query_states[start:end]
            query_ref_indices = query_index_tensor[start:end]

            state_distances = torch.cdist(
                query_chunk, reference_states, p=2
            )
            row_indices = torch.arange(
                end - start, dtype=torch.long, device=device
            )
            state_distances[row_indices, query_ref_indices] = float("inf")

            knn_distances, knn_indices = torch.topk(
                state_distances,
                k=KNN_SUPPORT_K,
                dim=1,
                largest=False,
                sorted=True,
            )
            neighbor_index_chunks.append(knn_indices)
            neighbor_distance_chunks.append(knn_distances)
            del state_distances

    neighbor_indices = torch.cat(neighbor_index_chunks, dim=0)
    neighbor_state_distances = torch.cat(neighbor_distance_chunks, dim=0)
    neighbor_actions = reference_actions[neighbor_indices]

    with torch.no_grad():
        behavior_local_sq_l2 = (
            query_behavior_actions.unsqueeze(1) - neighbor_actions
        ).pow(2).sum(dim=-1).min(dim=1).values

    behavior_local_sq_l2_cpu = (
        behavior_local_sq_l2.detach().cpu().numpy().astype(np.float64)
    )
    behavior_threshold = float(
        np.quantile(
            behavior_local_sq_l2_cpu,
            KNN_SUPPORT_BEHAVIOR_PERCENTILE / 100.0,
        )
    )
    if not np.isfinite(behavior_threshold):
        raise RuntimeError("kNN support produced a non-finite behavior threshold.")

    # Candidate actions are evaluated only on the support-query states, using
    # the raw environment observation (never the standardized kNN feature).
    query_raw_observations = torch.as_tensor(
        observations_cpu[query_indices], dtype=torch.float32, device=device
    )

    original_policy_state = state_dict_to_cpu(model.policy.state_dict())
    candidate_mean = []
    candidate_median = []
    candidate_p95 = []
    candidate_max = []
    candidate_unsupported_fraction = []
    candidate_mean_excess = []

    try:
        for candidate_index, agent in enumerate(agents):
            model.policy.load_state_dict(agent)
            model.policy.to(device)
            candidate_actions = _policy_actions_current_model(
                model, query_raw_observations
            )

            if candidate_actions.shape[0] != n_query:
                raise RuntimeError(
                    "kNN support candidate-action query count mismatch at "
                    f"candidate {candidate_index}."
                )

            with torch.no_grad():
                local_sq_l2 = (
                    candidate_actions.unsqueeze(1) - neighbor_actions
                ).pow(2).sum(dim=-1).min(dim=1).values

            local_cpu = local_sq_l2.detach().cpu().numpy().astype(np.float64)
            if not np.all(np.isfinite(local_cpu)):
                raise RuntimeError(
                    "kNN support produced non-finite local action distances "
                    f"for candidate {candidate_index}."
                )

            candidate_mean.append(float(np.mean(local_cpu)))
            candidate_median.append(float(np.median(local_cpu)))
            candidate_p95.append(float(np.quantile(local_cpu, 0.95)))
            candidate_max.append(float(np.max(local_cpu)))
            candidate_unsupported_fraction.append(
                float(np.mean(local_cpu > behavior_threshold))
            )
            candidate_mean_excess.append(
                float(np.mean(np.maximum(local_cpu - behavior_threshold, 0.0)))
            )
    finally:
        model.policy.load_state_dict(original_policy_state)
        model.policy.to(device)

    candidate_mean = np.asarray(candidate_mean, dtype=np.float64)
    candidate_median = np.asarray(candidate_median, dtype=np.float64)
    candidate_p95 = np.asarray(candidate_p95, dtype=np.float64)
    candidate_max = np.asarray(candidate_max, dtype=np.float64)
    candidate_unsupported_fraction = np.asarray(
        candidate_unsupported_fraction, dtype=np.float64
    )
    candidate_mean_excess = np.asarray(candidate_mean_excess, dtype=np.float64)

    behavior_mean = float(np.mean(behavior_local_sq_l2_cpu))
    behavior_median = float(np.median(behavior_local_sq_l2_cpu))
    behavior_p95 = float(np.quantile(behavior_local_sq_l2_cpu, 0.95))
    eps = 1e-12

    return {
        "reference_count": n_reference,
        "query_count": n_query,
        "query_indices": query_indices,
        "k": int(KNN_SUPPORT_K),
        "behavior_percentile": float(KNN_SUPPORT_BEHAVIOR_PERCENTILE),
        "behavior_threshold_sq_l2": behavior_threshold,
        "behavior_mean_sq_l2": behavior_mean,
        "behavior_median_sq_l2": behavior_median,
        "behavior_p95_sq_l2": behavior_p95,
        "mean_neighbor_state_distance": float(
            neighbor_state_distances.mean().detach().cpu().item()
        ),
        "p95_neighbor_state_distance": float(
            np.quantile(
                neighbor_state_distances.detach().cpu().numpy().reshape(-1),
                0.95,
            )
        ),
        "candidate_mean_sq_l2": candidate_mean,
        "candidate_median_sq_l2": candidate_median,
        "candidate_p95_sq_l2": candidate_p95,
        "candidate_max_sq_l2": candidate_max,
        "candidate_unsupported_fraction": candidate_unsupported_fraction,
        "candidate_mean_excess_sq_l2": candidate_mean_excess,
        "candidate_mean_ratio_to_behavior": candidate_mean / (behavior_mean + eps),
    }


def compute_state_conditional_knn_support_preserving_rng(
    model, agents, support_data
):
    """Run the kNN support estimator without perturbing experiment RNG state."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    try:
        return compute_state_conditional_knn_support(
            model, agents, support_data
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)



def compute_state_occupancy_knn_diagnostics(
    candidate_trajectory_episodes,
    replay_data,
):
    """Measure candidate state-occupancy novelty against frozen replay states.

    ``candidate_trajectory_episodes`` is a list with one entry per candidate;
    each candidate entry is a list of raw observation arrays, one array per
    already-completed online evaluation episode. These are diagnostics from the
    SAME trajectories used to produce ``cum_rews``. They are never inserted into
    replay and never used to fit FQE.

    The replay calibration is leave-one-out. For a deterministic subset of
    replay states, the query row itself is assigned infinite distance before
    retrieving the k nearest OTHER replay states. The k-th-neighbor radius is
    used rather than 1-NN distance because it is a local-density diagnostic and
    is less sensitive to a single accidental near-duplicate.

    When STATE_OCCUPANCY_INCLUDE_TIME=1, both replay and candidate points are
    represented as standardized [observation, t/H] features. Observation
    standardization and time standardization are fitted ONLY on frozen replay.
    """
    observations_cpu = np.asarray(
        replay_data["observations"], dtype=np.float32
    )
    if observations_cpu.ndim != 2:
        raise NotImplementedError(
            "State-occupancy kNN currently expects flat vector observations."
        )
    if observations_cpu.shape[0] <= STATE_OCCUPANCY_K:
        raise RuntimeError(
            "State-occupancy kNN requires more replay transitions than k: "
            f"reference={observations_cpu.shape[0]}, "
            f"k={STATE_OCCUPANCY_K}."
        )
    if not np.all(np.isfinite(observations_cpu)):
        raise RuntimeError(
            "State-occupancy replay observations contain non-finite values."
        )
    if len(candidate_trajectory_episodes) == 0:
        raise RuntimeError(
            "State-occupancy diagnostic received zero candidate trajectories."
        )

    n_reference = int(observations_cpu.shape[0])
    state_mean = observations_cpu.mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    state_std = observations_cpu.std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    state_std = np.where(
        state_std > 1e-6, state_std, 1.0
    ).astype(np.float32)

    standardized_replay = (
        (observations_cpu - state_mean) / state_std
    ).astype(np.float32, copy=False)

    finite_horizon_steps = replay_data.get("finite_horizon_steps", None)
    if STATE_OCCUPANCY_INCLUDE_TIME:
        if finite_horizon_steps is None:
            raise RuntimeError(
                "Time-aware state-occupancy kNN requires "
                "replay_data['finite_horizon_steps']."
            )
        finite_horizon_steps = int(finite_horizon_steps)
        if finite_horizon_steps <= 0:
            raise RuntimeError(
                "State-occupancy finite_horizon_steps must be > 0."
            )
        if "timesteps" not in replay_data:
            raise RuntimeError(
                "Time-aware state-occupancy kNN requires replay timesteps."
            )

        replay_timesteps = np.asarray(
            replay_data["timesteps"], dtype=np.float32
        ).reshape(-1)
        if len(replay_timesteps) != n_reference:
            raise RuntimeError(
                "State-occupancy replay observation/timestep length mismatch: "
                f"{n_reference} vs {len(replay_timesteps)}."
            )
        if (
            np.any(replay_timesteps < 0.0)
            or np.any(replay_timesteps >= float(finite_horizon_steps))
        ):
            raise RuntimeError(
                "State-occupancy replay timesteps are outside [0, H)."
            )

        replay_time_normalized = (
            replay_timesteps / float(finite_horizon_steps)
        ).astype(np.float32)
        time_mean = float(
            np.mean(replay_time_normalized, dtype=np.float64)
        )
        time_std = float(
            np.std(replay_time_normalized, dtype=np.float64)
        )
        if not np.isfinite(time_std) or time_std <= 1e-6:
            time_std = 1.0
        standardized_replay_time = (
            (replay_time_normalized - time_mean) / time_std
        ).astype(np.float32)
        replay_features_cpu = np.concatenate(
            (
                standardized_replay,
                standardized_replay_time.reshape(-1, 1),
            ),
            axis=1,
        )
    else:
        time_mean = 0.0
        time_std = 1.0
        replay_features_cpu = standardized_replay

    reference_features = torch.as_tensor(
        replay_features_cpu,
        dtype=torch.float32,
        device=device,
    )

    replay_query_indices = _deterministic_even_subsample_indices(
        n_reference,
        min(STATE_OCCUPANCY_REPLAY_QUERY_COUNT, n_reference),
    )
    replay_query_index_tensor = torch.as_tensor(
        replay_query_indices,
        dtype=torch.long,
        device=device,
    )
    replay_query_features = reference_features[replay_query_index_tensor]

    # Leave-one-out replay calibration.
    replay_knn_radius_chunks = []
    replay_1nn_chunks = []
    with torch.no_grad():
        for start in range(
            0,
            len(replay_query_indices),
            STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
        ):
            end = min(
                start + STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
                len(replay_query_indices),
            )
            query_chunk = replay_query_features[start:end]
            query_ref_indices = replay_query_index_tensor[start:end]

            distances = torch.cdist(
                query_chunk, reference_features, p=2
            )
            row_indices = torch.arange(
                end - start,
                dtype=torch.long,
                device=device,
            )
            distances[row_indices, query_ref_indices] = float("inf")

            local_distances = torch.topk(
                distances,
                k=STATE_OCCUPANCY_K,
                dim=1,
                largest=False,
                sorted=True,
            ).values
            replay_1nn_chunks.append(local_distances[:, 0])
            replay_knn_radius_chunks.append(local_distances[:, -1])
            del distances, local_distances

    replay_knn_radius = torch.cat(
        replay_knn_radius_chunks, dim=0
    ).detach().cpu().numpy().astype(np.float64)
    replay_1nn = torch.cat(
        replay_1nn_chunks, dim=0
    ).detach().cpu().numpy().astype(np.float64)

    behavior_threshold = float(
        np.quantile(
            replay_knn_radius,
            STATE_OCCUPANCY_BEHAVIOR_PERCENTILE / 100.0,
        )
    )
    if not np.isfinite(behavior_threshold):
        raise RuntimeError(
            "State-occupancy kNN produced a non-finite replay threshold."
        )

    candidate_total_states = []
    candidate_query_states = []
    candidate_mean_knn_radius = []
    candidate_median_knn_radius = []
    candidate_p95_knn_radius = []
    candidate_max_knn_radius = []
    candidate_mean_1nn_distance = []
    candidate_p95_1nn_distance = []
    candidate_ood_fraction = []
    candidate_mean_excess = []

    for candidate_idx, episodes in enumerate(candidate_trajectory_episodes):
        if not isinstance(episodes, (list, tuple)) or len(episodes) == 0:
            raise RuntimeError(
                "State-occupancy diagnostic is missing completed episodes for "
                f"candidate {candidate_idx}."
            )

        candidate_states_parts = []
        candidate_timesteps_parts = []
        for episode_idx, episode_states in enumerate(episodes):
            episode_states = np.asarray(
                episode_states, dtype=np.float32
            )
            if episode_states.ndim != 2:
                raise RuntimeError(
                    "State-occupancy candidate episode must have shape "
                    f"[T, obs_dim], got {episode_states.shape} for candidate "
                    f"{candidate_idx}, episode {episode_idx}."
                )
            if episode_states.shape[1] != observations_cpu.shape[1]:
                raise RuntimeError(
                    "State-occupancy candidate/replay observation dimension "
                    f"mismatch: candidate={episode_states.shape[1]}, "
                    f"replay={observations_cpu.shape[1]}."
                )
            if episode_states.shape[0] <= 0:
                raise RuntimeError(
                    "State-occupancy diagnostic encountered an empty candidate "
                    f"episode for candidate {candidate_idx}."
                )
            if not np.all(np.isfinite(episode_states)):
                raise RuntimeError(
                    "State-occupancy candidate trajectory contains non-finite "
                    f"states for candidate {candidate_idx}."
                )
            if (
                STATE_OCCUPANCY_INCLUDE_TIME
                and episode_states.shape[0] > finite_horizon_steps
            ):
                raise RuntimeError(
                    "Candidate online episode exceeds the finite horizon used "
                    f"by FQE: length={episode_states.shape[0]}, "
                    f"H={finite_horizon_steps}."
                )

            candidate_states_parts.append(episode_states)
            candidate_timesteps_parts.append(
                np.arange(
                    episode_states.shape[0], dtype=np.float32
                )
            )

        candidate_states = np.concatenate(
            candidate_states_parts, axis=0
        )
        candidate_timesteps = np.concatenate(
            candidate_timesteps_parts, axis=0
        )
        total_states = int(candidate_states.shape[0])

        candidate_indices = _deterministic_even_subsample_indices(
            total_states,
            min(STATE_OCCUPANCY_CANDIDATE_QUERY_COUNT, total_states),
        )
        query_states = candidate_states[candidate_indices]
        standardized_query_states = (
            (query_states - state_mean) / state_std
        ).astype(np.float32, copy=False)

        if STATE_OCCUPANCY_INCLUDE_TIME:
            query_t = candidate_timesteps[candidate_indices]
            query_t_normalized = (
                query_t / float(finite_horizon_steps)
            ).astype(np.float32)
            standardized_query_t = (
                (query_t_normalized - time_mean) / time_std
            ).astype(np.float32)
            query_features_cpu = np.concatenate(
                (
                    standardized_query_states,
                    standardized_query_t.reshape(-1, 1),
                ),
                axis=1,
            )
        else:
            query_features_cpu = standardized_query_states

        query_features = torch.as_tensor(
            query_features_cpu,
            dtype=torch.float32,
            device=device,
        )

        candidate_radius_chunks = []
        candidate_1nn_chunks = []
        with torch.no_grad():
            for start in range(
                0,
                len(candidate_indices),
                STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
            ):
                end = min(
                    start + STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
                    len(candidate_indices),
                )
                distances = torch.cdist(
                    query_features[start:end],
                    reference_features,
                    p=2,
                )
                local_distances = torch.topk(
                    distances,
                    k=STATE_OCCUPANCY_K,
                    dim=1,
                    largest=False,
                    sorted=True,
                ).values
                candidate_1nn_chunks.append(local_distances[:, 0])
                candidate_radius_chunks.append(local_distances[:, -1])
                del distances, local_distances

        candidate_radius = torch.cat(
            candidate_radius_chunks, dim=0
        ).detach().cpu().numpy().astype(np.float64)
        candidate_1nn = torch.cat(
            candidate_1nn_chunks, dim=0
        ).detach().cpu().numpy().astype(np.float64)

        candidate_total_states.append(total_states)
        candidate_query_states.append(int(len(candidate_indices)))
        candidate_mean_knn_radius.append(float(np.mean(candidate_radius)))
        candidate_median_knn_radius.append(
            float(np.median(candidate_radius))
        )
        candidate_p95_knn_radius.append(
            float(np.quantile(candidate_radius, 0.95))
        )
        candidate_max_knn_radius.append(float(np.max(candidate_radius)))
        candidate_mean_1nn_distance.append(float(np.mean(candidate_1nn)))
        candidate_p95_1nn_distance.append(
            float(np.quantile(candidate_1nn, 0.95))
        )
        candidate_ood_fraction.append(
            float(np.mean(candidate_radius > behavior_threshold))
        )
        candidate_mean_excess.append(
            float(
                np.mean(
                    np.maximum(
                        candidate_radius - behavior_threshold,
                        0.0,
                    )
                )
            )
        )

    candidate_mean_knn_radius = np.asarray(
        candidate_mean_knn_radius, dtype=np.float64
    )
    behavior_mean_radius = float(np.mean(replay_knn_radius))
    eps = 1e-12

    return {
        "reference_count": int(n_reference),
        "replay_query_count": int(len(replay_query_indices)),
        "k": int(STATE_OCCUPANCY_K),
        "behavior_percentile": float(
            STATE_OCCUPANCY_BEHAVIOR_PERCENTILE
        ),
        "behavior_threshold_knn_radius": behavior_threshold,
        "behavior_mean_knn_radius": behavior_mean_radius,
        "behavior_median_knn_radius": float(
            np.median(replay_knn_radius)
        ),
        "behavior_p95_knn_radius": float(
            np.quantile(replay_knn_radius, 0.95)
        ),
        "behavior_mean_1nn_distance": float(np.mean(replay_1nn)),
        "include_time": bool(STATE_OCCUPANCY_INCLUDE_TIME),
        "finite_horizon_steps": (
            int(finite_horizon_steps)
            if finite_horizon_steps is not None
            else -1
        ),
        "candidate_total_states": np.asarray(
            candidate_total_states, dtype=np.int64
        ),
        "candidate_query_states": np.asarray(
            candidate_query_states, dtype=np.int64
        ),
        "candidate_mean_knn_radius": candidate_mean_knn_radius,
        "candidate_median_knn_radius": np.asarray(
            candidate_median_knn_radius, dtype=np.float64
        ),
        "candidate_p95_knn_radius": np.asarray(
            candidate_p95_knn_radius, dtype=np.float64
        ),
        "candidate_max_knn_radius": np.asarray(
            candidate_max_knn_radius, dtype=np.float64
        ),
        "candidate_mean_1nn_distance": np.asarray(
            candidate_mean_1nn_distance, dtype=np.float64
        ),
        "candidate_p95_1nn_distance": np.asarray(
            candidate_p95_1nn_distance, dtype=np.float64
        ),
        "candidate_ood_fraction": np.asarray(
            candidate_ood_fraction, dtype=np.float64
        ),
        "candidate_mean_excess_knn_radius": np.asarray(
            candidate_mean_excess, dtype=np.float64
        ),
        "candidate_mean_ratio_to_behavior": (
            candidate_mean_knn_radius / (behavior_mean_radius + eps)
        ),
    }


def compute_state_occupancy_fqe_metrics(
    online_scores,
    raw_fqe_scores,
    occupancy_diagnostics,
    iteration,
    score_metadata=None,
):
    """Relate online occupancy novelty to FQE ranking error.

    Online states are used ONLY after their returns have already been measured.
    Nothing returned here participates in PPO/ESA selection or FQE fitting.
    """
    online_scores = np.asarray(online_scores, dtype=np.float64)
    raw_fqe_scores = np.asarray(raw_fqe_scores, dtype=np.float64)
    mean_radius = np.asarray(
        occupancy_diagnostics["candidate_mean_knn_radius"],
        dtype=np.float64,
    )
    ood_fraction = np.asarray(
        occupancy_diagnostics["candidate_ood_fraction"],
        dtype=np.float64,
    )

    if not (
        online_scores.shape
        == raw_fqe_scores.shape
        == mean_radius.shape
        == ood_fraction.shape
    ):
        raise RuntimeError(
            "State-occupancy metric length mismatch: "
            f"online={online_scores.shape}, FQE={raw_fqe_scores.shape}, "
            f"radius={mean_radius.shape}, OOD={ood_fraction.shape}."
        )
    if len(online_scores) == 0:
        raise RuntimeError(
            "State-occupancy FQE metrics received zero candidates."
        )
    if not (
        np.all(np.isfinite(online_scores))
        and np.all(np.isfinite(raw_fqe_scores))
        and np.all(np.isfinite(mean_radius))
        and np.all(np.isfinite(ood_fraction))
    ):
        raise RuntimeError(
            "State-occupancy FQE metrics received non-finite values."
        )

    n_candidates = int(len(online_scores))
    online_order = np.argsort(online_scores)[::-1]
    fqe_order = np.argsort(raw_fqe_scores)[::-1]

    online_rank = np.empty(n_candidates, dtype=np.int64)
    online_rank[online_order] = np.arange(
        1, n_candidates + 1, dtype=np.int64
    )
    fqe_rank = np.empty(n_candidates, dtype=np.int64)
    fqe_rank[fqe_order] = np.arange(
        1, n_candidates + 1, dtype=np.int64
    )
    abs_rank_error = np.abs(fqe_rank - online_rank).astype(np.float64)

    # Rank smaller mean kNN radius as lower occupancy novelty.  OOD fraction
    # is only a deterministic tie-breaker, so the reported occupancy rank and
    # low/high-novelty quartiles match the continuous novelty quantity used by
    # the main occupancy-vs-FQE-error correlations.
    occupancy_order = np.lexsort((ood_fraction, mean_radius))
    occupancy_rank = np.empty(n_candidates, dtype=np.int64)
    occupancy_rank[occupancy_order] = np.arange(
        1, n_candidates + 1, dtype=np.int64
    )

    def _safe_standardize(values):
        values = np.asarray(values, dtype=np.float64)
        scale = float(np.std(values, ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            return np.zeros_like(values)
        return (values - float(np.mean(values))) / scale

    online_z = _safe_standardize(online_scores)
    fqe_z = _safe_standardize(raw_fqe_scores)
    abs_z_error = np.abs(fqe_z - online_z)

    def _corr(x, y, method="spearman"):
        frame = pd.DataFrame({"x": x, "y": y})
        return float(frame.corr(method=method).loc["x", "y"])

    oracle_idx = int(np.argmax(online_scores))
    fqe_idx = int(np.argmax(raw_fqe_scores))

    quartile_size = max(1, int(math.ceil(n_candidates / 4.0)))
    low_novelty = occupancy_order[:quartile_size]
    high_novelty = occupancy_order[-quartile_size:]

    metadata = score_metadata if score_metadata is not None else object()
    finite_horizon_steps = getattr(metadata, "finite_horizon_steps", None)

    metrics = {
        "iteration": int(iteration),
        "fqe_config": (
            f"{int(getattr(metadata, 'fqe_n_steps', STATE_OCCUPANCY_FQE_N_STEPS))}"
            f"_steps_target"
            f"{int(getattr(metadata, 'fqe_target_update_interval', STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL))}"
        ),
        "fqe_n_steps": int(
            getattr(
                metadata,
                "fqe_n_steps",
                STATE_OCCUPANCY_FQE_N_STEPS,
            )
        ),
        "fqe_target_update_interval": int(
            getattr(
                metadata,
                "fqe_target_update_interval",
                STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL,
            )
        ),
        "fqe_objective": str(
            getattr(metadata, "fqe_objective", "unknown")
        ),
        "fqe_gamma": float(getattr(metadata, "fqe_gamma", np.nan)),
        "finite_horizon_steps": (
            -1 if finite_horizon_steps is None
            else int(finite_horizon_steps)
        ),
        "time_conditioned": bool(
            getattr(metadata, "time_conditioned", False)
        ),
        "reference_transitions": int(
            occupancy_diagnostics["reference_count"]
        ),
        "replay_query_states": int(
            occupancy_diagnostics["replay_query_count"]
        ),
        "knn_k": int(occupancy_diagnostics["k"]),
        "include_time": bool(occupancy_diagnostics["include_time"]),
        "behavior_percentile": float(
            occupancy_diagnostics["behavior_percentile"]
        ),
        "behavior_threshold_knn_radius": float(
            occupancy_diagnostics["behavior_threshold_knn_radius"]
        ),
        "behavior_mean_knn_radius": float(
            occupancy_diagnostics["behavior_mean_knn_radius"]
        ),
        "mean_candidate_knn_radius": float(np.mean(mean_radius)),
        "mean_candidate_ood_fraction": float(np.mean(ood_fraction)),
        "max_candidate_ood_fraction": float(np.max(ood_fraction)),
        "occupancy_vs_online_spearman": _corr(
            mean_radius, online_scores
        ),
        "occupancy_ood_vs_online_spearman": _corr(
            ood_fraction, online_scores
        ),
        "occupancy_vs_fqe_spearman": _corr(
            mean_radius, raw_fqe_scores
        ),
        # Positive values here support the hypothesis that stronger occupancy
        # shift is associated with larger FQE ranking/calibration error.
        "occupancy_vs_abs_fqe_rank_error_spearman": _corr(
            mean_radius, abs_rank_error
        ),
        "occupancy_ood_vs_abs_fqe_rank_error_spearman": _corr(
            ood_fraction, abs_rank_error
        ),
        "occupancy_vs_abs_fqe_z_error_spearman": _corr(
            mean_radius, abs_z_error
        ),
        "oracle_idx": oracle_idx,
        "oracle_occupancy_rank": int(occupancy_rank[oracle_idx]),
        "oracle_mean_knn_radius": float(mean_radius[oracle_idx]),
        "oracle_ood_fraction": float(ood_fraction[oracle_idx]),
        "fqe_idx": fqe_idx,
        "fqe_selected_occupancy_rank": int(occupancy_rank[fqe_idx]),
        "fqe_selected_mean_knn_radius": float(mean_radius[fqe_idx]),
        "fqe_selected_ood_fraction": float(ood_fraction[fqe_idx]),
        "mean_abs_fqe_rank_error": float(np.mean(abs_rank_error)),
        "low_novelty_quartile_mean_abs_rank_error": float(
            np.mean(abs_rank_error[low_novelty])
        ),
        "high_novelty_quartile_mean_abs_rank_error": float(
            np.mean(abs_rank_error[high_novelty])
        ),
        "high_minus_low_novelty_rank_error": float(
            np.mean(abs_rank_error[high_novelty])
            - np.mean(abs_rank_error[low_novelty])
        ),
    }

    candidate_details = {
        "online_rank": online_rank,
        "fqe_rank": fqe_rank,
        "abs_fqe_rank_error": abs_rank_error,
        "abs_fqe_z_error": abs_z_error,
        "occupancy_rank": occupancy_rank,
    }
    return metrics, candidate_details



def compute_time_resolved_state_occupancy_diagnostics(
    candidate_trajectory_episodes,
    replay_data,
    online_scores,
    raw_fqe_scores,
    iteration,
    score_metadata=None,
):
    """Resolve WHEN candidate occupancy leaves replay support.

    The candidate trajectories are the same read-only online traces already
    used by the whole-trajectory occupancy diagnostic. This function performs
    no rollouts, does not load candidate policies, does not mutate replay, and
    does not train or alter FQE.

    For every configured normalized time window:
      * restrict the replay reference to states whose reconstructed timestep is
        inside that same window;
      * calibrate a leave-one-out k-th-neighbor radius threshold on that
        window's replay states;
      * compute candidate kNN radius/OOD statistics using only candidate states
        visited in the same window;
      * relate window novelty to online return and to both absolute and SIGNED
        FQE ranking/calibration error.

    Positive ``fqe_rank_overvaluation`` means FQE ranks a candidate too highly:
        online_rank - fqe_rank > 0.
    Positive ``fqe_z_overvaluation`` means the standardized FQE value is higher
    than the standardized online value.
    """
    online_scores = np.asarray(online_scores, dtype=np.float64)
    raw_fqe_scores = np.asarray(raw_fqe_scores, dtype=np.float64)
    if online_scores.shape != raw_fqe_scores.shape:
        raise RuntimeError(
            "Time-resolved occupancy online/FQE length mismatch: "
            f"{online_scores.shape} vs {raw_fqe_scores.shape}."
        )
    n_candidates = int(len(online_scores))
    if n_candidates == 0:
        raise RuntimeError(
            "Time-resolved occupancy diagnostic received zero candidates."
        )
    if len(candidate_trajectory_episodes) != n_candidates:
        raise RuntimeError(
            "Time-resolved occupancy trajectory count mismatch: "
            f"{len(candidate_trajectory_episodes)} trajectories for "
            f"{n_candidates} candidates."
        )
    if not (
        np.all(np.isfinite(online_scores))
        and np.all(np.isfinite(raw_fqe_scores))
    ):
        raise RuntimeError(
            "Time-resolved occupancy received non-finite online/FQE scores."
        )

    observations_cpu = np.asarray(
        replay_data["observations"], dtype=np.float32
    )
    replay_timesteps = np.asarray(
        replay_data.get("timesteps", []), dtype=np.float32
    ).reshape(-1)
    finite_horizon_steps = replay_data.get("finite_horizon_steps", None)

    if observations_cpu.ndim != 2:
        raise NotImplementedError(
            "Time-resolved occupancy currently expects flat vector observations."
        )
    if finite_horizon_steps is None:
        raise RuntimeError(
            "Time-resolved occupancy requires a finite_horizon_steps value."
        )
    finite_horizon_steps = int(finite_horizon_steps)
    if finite_horizon_steps <= 0:
        raise RuntimeError(
            "Time-resolved occupancy finite_horizon_steps must be > 0."
        )
    if len(replay_timesteps) != len(observations_cpu):
        raise RuntimeError(
            "Time-resolved occupancy replay observation/timestep mismatch: "
            f"{len(observations_cpu)} vs {len(replay_timesteps)}."
        )
    if not np.all(np.isfinite(observations_cpu)):
        raise RuntimeError(
            "Time-resolved occupancy replay observations contain non-finite values."
        )
    if (
        np.any(replay_timesteps < 0.0)
        or np.any(replay_timesteps >= float(finite_horizon_steps))
    ):
        raise RuntimeError(
            "Time-resolved occupancy replay timesteps are outside [0, H)."
        )

    # Resolve normalized boundaries to integer environment timesteps.
    boundaries = [
        int(round(float(frac) * finite_horizon_steps))
        for frac in TIME_RESOLVED_OCCUPANCY_BOUNDARIES
    ]
    boundaries[0] = 0
    boundaries[-1] = finite_horizon_steps
    if any(
        right <= left
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ):
        raise RuntimeError(
            "Resolved time-resolved occupancy windows are empty/overlapping. "
            f"H={finite_horizon_steps}, boundaries={boundaries}."
        )

    # Keep the SAME feature scaling as the whole-trajectory occupancy study:
    # fit state/time normalization once on the full frozen replay distribution.
    state_mean = observations_cpu.mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    state_std = observations_cpu.std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    state_std = np.where(
        state_std > 1e-6, state_std, 1.0
    ).astype(np.float32)
    standardized_replay = (
        (observations_cpu - state_mean) / state_std
    ).astype(np.float32, copy=False)

    if STATE_OCCUPANCY_INCLUDE_TIME:
        replay_time_normalized = (
            replay_timesteps / float(finite_horizon_steps)
        ).astype(np.float32)
        time_mean = float(
            np.mean(replay_time_normalized, dtype=np.float64)
        )
        time_std = float(
            np.std(replay_time_normalized, dtype=np.float64)
        )
        if not np.isfinite(time_std) or time_std <= 1e-6:
            time_std = 1.0
        standardized_replay_time = (
            (replay_time_normalized - time_mean) / time_std
        ).astype(np.float32)
        replay_features_cpu = np.concatenate(
            (
                standardized_replay,
                standardized_replay_time.reshape(-1, 1),
            ),
            axis=1,
        )
    else:
        time_mean = 0.0
        time_std = 1.0
        replay_features_cpu = standardized_replay

    # Candidate-wide FQE error annotations are identical for every time window.
    online_order = np.argsort(online_scores)[::-1]
    fqe_order = np.argsort(raw_fqe_scores)[::-1]
    online_rank = np.empty(n_candidates, dtype=np.int64)
    online_rank[online_order] = np.arange(
        1, n_candidates + 1, dtype=np.int64
    )
    fqe_rank = np.empty(n_candidates, dtype=np.int64)
    fqe_rank[fqe_order] = np.arange(
        1, n_candidates + 1, dtype=np.int64
    )
    fqe_rank_overvaluation = (
        online_rank.astype(np.float64) - fqe_rank.astype(np.float64)
    )
    abs_rank_error = np.abs(fqe_rank_overvaluation)

    def _standardize(values):
        values = np.asarray(values, dtype=np.float64)
        scale = float(np.std(values, ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            return np.zeros_like(values)
        return (values - float(np.mean(values))) / scale

    online_z = _standardize(online_scores)
    fqe_z = _standardize(raw_fqe_scores)
    fqe_z_overvaluation = fqe_z - online_z
    abs_z_error = np.abs(fqe_z_overvaluation)

    def _safe_spearman(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(valid)) < 3:
            return float("nan")
        x_valid = x[valid]
        y_valid = y[valid]
        if (
            float(np.ptp(x_valid)) <= 1e-12
            or float(np.ptp(y_valid)) <= 1e-12
        ):
            return float("nan")
        frame = pd.DataFrame({"x": x_valid, "y": y_valid})
        return float(frame.corr(method="spearman").loc["x", "y"])

    metadata = score_metadata if score_metadata is not None else object()
    fqe_n_steps = int(
        getattr(metadata, "fqe_n_steps", STATE_OCCUPANCY_FQE_N_STEPS)
    )
    fqe_target_interval = int(
        getattr(
            metadata,
            "fqe_target_update_interval",
            STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL,
        )
    )

    # Validate and cache raw candidate episodes once. Timesteps always restart
    # from zero at each already-completed online evaluation episode.
    candidate_episode_cache = []
    for candidate_idx, episodes in enumerate(candidate_trajectory_episodes):
        if not isinstance(episodes, (list, tuple)) or len(episodes) == 0:
            raise RuntimeError(
                "Time-resolved occupancy is missing completed episodes for "
                f"candidate {candidate_idx}."
            )
        cached_episodes = []
        for episode_idx, episode_states in enumerate(episodes):
            episode_states = np.asarray(
                episode_states, dtype=np.float32
            )
            if (
                episode_states.ndim != 2
                or episode_states.shape[1] != observations_cpu.shape[1]
            ):
                raise RuntimeError(
                    "Time-resolved occupancy candidate episode shape mismatch "
                    f"for candidate {candidate_idx}, episode {episode_idx}: "
                    f"{episode_states.shape}."
                )
            if episode_states.shape[0] <= 0:
                raise RuntimeError(
                    "Time-resolved occupancy encountered an empty candidate "
                    f"episode for candidate {candidate_idx}."
                )
            if episode_states.shape[0] > finite_horizon_steps:
                raise RuntimeError(
                    "Time-resolved occupancy candidate episode exceeds H: "
                    f"length={episode_states.shape[0]}, "
                    f"H={finite_horizon_steps}."
                )
            if not np.all(np.isfinite(episode_states)):
                raise RuntimeError(
                    "Time-resolved occupancy candidate trajectory contains "
                    f"non-finite states for candidate {candidate_idx}."
                )
            cached_episodes.append(episode_states)
        candidate_episode_cache.append(cached_episodes)

    summary_rows = []
    candidate_rows = []
    oracle_idx = int(np.argmax(online_scores))
    fqe_idx = int(np.argmax(raw_fqe_scores))

    for window_index, (window_start, window_end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        window_label = f"{window_start}_{window_end}"

        replay_window_mask = (
            (replay_timesteps >= float(window_start))
            & (replay_timesteps < float(window_end))
        )
        replay_window_indices = np.flatnonzero(replay_window_mask)
        n_window_reference = int(len(replay_window_indices))
        if n_window_reference <= STATE_OCCUPANCY_K:
            raise RuntimeError(
                "Time-resolved occupancy window has too few replay states for "
                f"kNN: window=[{window_start},{window_end}), "
                f"reference={n_window_reference}, k={STATE_OCCUPANCY_K}."
            )

        window_reference_features = torch.as_tensor(
            replay_features_cpu[replay_window_indices],
            dtype=torch.float32,
            device=device,
        )

        # Leave-one-out calibration inside THIS time window.
        replay_query_local_indices = _deterministic_even_subsample_indices(
            n_window_reference,
            min(
                TIME_RESOLVED_OCCUPANCY_REPLAY_QUERY_COUNT_PER_WINDOW,
                n_window_reference,
            ),
        )
        replay_query_local_tensor = torch.as_tensor(
            replay_query_local_indices,
            dtype=torch.long,
            device=device,
        )
        replay_query_features = window_reference_features[
            replay_query_local_tensor
        ]

        replay_radius_chunks = []
        replay_1nn_chunks = []
        with torch.no_grad():
            for start in range(
                0,
                len(replay_query_local_indices),
                STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
            ):
                end = min(
                    start + STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
                    len(replay_query_local_indices),
                )
                distances = torch.cdist(
                    replay_query_features[start:end],
                    window_reference_features,
                    p=2,
                )
                rows = torch.arange(
                    end - start, dtype=torch.long, device=device
                )
                self_local = replay_query_local_tensor[start:end]
                distances[rows, self_local] = float("inf")
                local_distances = torch.topk(
                    distances,
                    k=STATE_OCCUPANCY_K,
                    dim=1,
                    largest=False,
                    sorted=True,
                ).values
                replay_1nn_chunks.append(local_distances[:, 0])
                replay_radius_chunks.append(local_distances[:, -1])
                del distances, local_distances

        replay_radius = torch.cat(
            replay_radius_chunks, dim=0
        ).detach().cpu().numpy().astype(np.float64)
        replay_1nn = torch.cat(
            replay_1nn_chunks, dim=0
        ).detach().cpu().numpy().astype(np.float64)

        behavior_threshold = float(
            np.quantile(
                replay_radius,
                STATE_OCCUPANCY_BEHAVIOR_PERCENTILE / 100.0,
            )
        )
        behavior_mean_radius = float(np.mean(replay_radius))
        if (
            not np.isfinite(behavior_threshold)
            or not np.isfinite(behavior_mean_radius)
        ):
            raise RuntimeError(
                "Time-resolved occupancy produced non-finite replay "
                f"calibration in window {window_label}."
            )

        cand_total_states = np.zeros(n_candidates, dtype=np.int64)
        cand_query_states = np.zeros(n_candidates, dtype=np.int64)
        cand_mean_radius = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_median_radius = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_p95_radius = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_max_radius = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_mean_1nn = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_p95_1nn = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_ood_fraction = np.full(n_candidates, np.nan, dtype=np.float64)
        cand_mean_excess = np.full(n_candidates, np.nan, dtype=np.float64)

        for candidate_idx, episodes in enumerate(candidate_episode_cache):
            window_states_parts = []
            window_t_parts = []
            for episode_states in episodes:
                episode_length = int(episode_states.shape[0])
                local_start = min(window_start, episode_length)
                local_end = min(window_end, episode_length)
                if local_start >= local_end:
                    continue
                window_states_parts.append(
                    episode_states[local_start:local_end]
                )
                window_t_parts.append(
                    np.arange(
                        local_start, local_end, dtype=np.float32
                    )
                )

            if not window_states_parts:
                # Early termination can legitimately leave a late window empty.
                # Keep NaNs for the occupancy values and record zero states.
                continue

            window_states = np.concatenate(window_states_parts, axis=0)
            window_t = np.concatenate(window_t_parts, axis=0)
            cand_total_states[candidate_idx] = int(len(window_states))

            candidate_query_indices = _deterministic_even_subsample_indices(
                len(window_states),
                min(
                    TIME_RESOLVED_OCCUPANCY_CANDIDATE_QUERY_COUNT_PER_WINDOW,
                    len(window_states),
                ),
            )
            query_states = window_states[candidate_query_indices]
            query_t = window_t[candidate_query_indices]
            cand_query_states[candidate_idx] = int(len(candidate_query_indices))

            standardized_query_states = (
                (query_states - state_mean) / state_std
            ).astype(np.float32, copy=False)

            if STATE_OCCUPANCY_INCLUDE_TIME:
                query_t_normalized = (
                    query_t / float(finite_horizon_steps)
                ).astype(np.float32)
                standardized_query_t = (
                    (query_t_normalized - time_mean) / time_std
                ).astype(np.float32)
                query_features_cpu = np.concatenate(
                    (
                        standardized_query_states,
                        standardized_query_t.reshape(-1, 1),
                    ),
                    axis=1,
                )
            else:
                query_features_cpu = standardized_query_states

            query_features = torch.as_tensor(
                query_features_cpu,
                dtype=torch.float32,
                device=device,
            )

            candidate_radius_chunks = []
            candidate_1nn_chunks = []
            with torch.no_grad():
                for start in range(
                    0,
                    len(candidate_query_indices),
                    STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
                ):
                    end = min(
                        start + STATE_OCCUPANCY_QUERY_CHUNK_SIZE,
                        len(candidate_query_indices),
                    )
                    distances = torch.cdist(
                        query_features[start:end],
                        window_reference_features,
                        p=2,
                    )
                    local_distances = torch.topk(
                        distances,
                        k=STATE_OCCUPANCY_K,
                        dim=1,
                        largest=False,
                        sorted=True,
                    ).values
                    candidate_1nn_chunks.append(local_distances[:, 0])
                    candidate_radius_chunks.append(local_distances[:, -1])
                    del distances, local_distances

            candidate_radius = torch.cat(
                candidate_radius_chunks, dim=0
            ).detach().cpu().numpy().astype(np.float64)
            candidate_1nn = torch.cat(
                candidate_1nn_chunks, dim=0
            ).detach().cpu().numpy().astype(np.float64)

            cand_mean_radius[candidate_idx] = float(np.mean(candidate_radius))
            cand_median_radius[candidate_idx] = float(
                np.median(candidate_radius)
            )
            cand_p95_radius[candidate_idx] = float(
                np.quantile(candidate_radius, 0.95)
            )
            cand_max_radius[candidate_idx] = float(np.max(candidate_radius))
            cand_mean_1nn[candidate_idx] = float(np.mean(candidate_1nn))
            cand_p95_1nn[candidate_idx] = float(
                np.quantile(candidate_1nn, 0.95)
            )
            cand_ood_fraction[candidate_idx] = float(
                np.mean(candidate_radius > behavior_threshold)
            )
            cand_mean_excess[candidate_idx] = float(
                np.mean(
                    np.maximum(candidate_radius - behavior_threshold, 0.0)
                )
            )

        valid_candidates = np.isfinite(cand_mean_radius)
        n_valid_candidates = int(np.sum(valid_candidates))
        if n_valid_candidates == 0:
            raise RuntimeError(
                "Time-resolved occupancy found no candidate states in window "
                f"{window_label}."
            )

        cand_ratio = cand_mean_radius / (behavior_mean_radius + 1e-12)
        occupancy_rank = np.full(n_candidates, -1, dtype=np.int64)
        valid_indices = np.flatnonzero(valid_candidates)
        valid_order = valid_indices[
            np.lexsort(
                (
                    cand_ood_fraction[valid_indices],
                    cand_mean_radius[valid_indices],
                )
            )
        ]
        occupancy_rank[valid_order] = np.arange(
            1, len(valid_order) + 1, dtype=np.int64
        )

        summary_rows.append({
            "iteration": int(iteration),
            "window_index": int(window_index),
            "window_label": window_label,
            "window_start": int(window_start),
            "window_end": int(window_end),
            "window_start_fraction": float(
                window_start / float(finite_horizon_steps)
            ),
            "window_end_fraction": float(
                window_end / float(finite_horizon_steps)
            ),
            "window_width": int(window_end - window_start),
            "fqe_config": f"{fqe_n_steps}_steps_target{fqe_target_interval}",
            "fqe_n_steps": int(fqe_n_steps),
            "fqe_target_update_interval": int(fqe_target_interval),
            "fqe_objective": str(
                getattr(metadata, "fqe_objective", "unknown")
            ),
            "fqe_gamma": float(getattr(metadata, "fqe_gamma", np.nan)),
            "finite_horizon_steps": int(finite_horizon_steps),
            "time_conditioned": bool(
                getattr(metadata, "time_conditioned", False)
            ),
            "include_time": bool(STATE_OCCUPANCY_INCLUDE_TIME),
            "knn_k": int(STATE_OCCUPANCY_K),
            "behavior_percentile": float(
                STATE_OCCUPANCY_BEHAVIOR_PERCENTILE
            ),
            "reference_transitions_window": int(n_window_reference),
            "replay_query_states_window": int(
                len(replay_query_local_indices)
            ),
            "behavior_threshold_knn_radius": float(behavior_threshold),
            "behavior_mean_knn_radius": float(behavior_mean_radius),
            "behavior_p95_knn_radius": float(
                np.quantile(replay_radius, 0.95)
            ),
            "behavior_mean_1nn_distance": float(np.mean(replay_1nn)),
            "candidates_with_states": int(n_valid_candidates),
            "candidate_presence_fraction": float(
                n_valid_candidates / n_candidates
            ),
            "mean_candidate_window_states": float(
                np.mean(cand_total_states)
            ),
            "mean_candidate_query_states": float(
                np.mean(cand_query_states)
            ),
            "mean_candidate_knn_radius": float(
                np.nanmean(cand_mean_radius)
            ),
            "mean_candidate_radius_ratio_to_behavior": float(
                np.nanmean(cand_ratio)
            ),
            "mean_candidate_ood_fraction": float(
                np.nanmean(cand_ood_fraction)
            ),
            "max_candidate_ood_fraction": float(
                np.nanmax(cand_ood_fraction)
            ),
            "occupancy_vs_online_spearman": _safe_spearman(
                cand_mean_radius, online_scores
            ),
            "occupancy_ood_vs_online_spearman": _safe_spearman(
                cand_ood_fraction, online_scores
            ),
            "occupancy_vs_fqe_spearman": _safe_spearman(
                cand_mean_radius, raw_fqe_scores
            ),
            "occupancy_vs_abs_fqe_rank_error_spearman": _safe_spearman(
                cand_mean_radius, abs_rank_error
            ),
            "occupancy_ood_vs_abs_fqe_rank_error_spearman": _safe_spearman(
                cand_ood_fraction, abs_rank_error
            ),
            # Positive correlation here means more-novel states are
            # systematically OVERVALUED by FQE relative to their online rank.
            "occupancy_vs_fqe_rank_overvaluation_spearman": _safe_spearman(
                cand_mean_radius, fqe_rank_overvaluation
            ),
            "occupancy_ood_vs_fqe_rank_overvaluation_spearman": _safe_spearman(
                cand_ood_fraction, fqe_rank_overvaluation
            ),
            "occupancy_vs_abs_fqe_z_error_spearman": _safe_spearman(
                cand_mean_radius, abs_z_error
            ),
            "occupancy_vs_fqe_z_overvaluation_spearman": _safe_spearman(
                cand_mean_radius, fqe_z_overvaluation
            ),
            "oracle_idx": int(oracle_idx),
            "oracle_occupancy_rank": int(occupancy_rank[oracle_idx]),
            "oracle_mean_knn_radius": float(cand_mean_radius[oracle_idx]),
            "oracle_ood_fraction": float(cand_ood_fraction[oracle_idx]),
            "fqe_idx": int(fqe_idx),
            "fqe_selected_occupancy_rank": int(
                occupancy_rank[fqe_idx]
            ),
            "fqe_selected_mean_knn_radius": float(
                cand_mean_radius[fqe_idx]
            ),
            "fqe_selected_ood_fraction": float(
                cand_ood_fraction[fqe_idx]
            ),
        })

        for candidate_idx in range(n_candidates):
            candidate_rows.append({
                "iteration": int(iteration),
                "candidate": int(candidate_idx),
                "window_index": int(window_index),
                "window_label": window_label,
                "window_start": int(window_start),
                "window_end": int(window_end),
                "online": float(online_scores[candidate_idx]),
                "fqe_mean_q": float(raw_fqe_scores[candidate_idx]),
                "online_rank": int(online_rank[candidate_idx]),
                "fqe_rank": int(fqe_rank[candidate_idx]),
                "fqe_rank_overvaluation": float(
                    fqe_rank_overvaluation[candidate_idx]
                ),
                "abs_fqe_rank_error": float(
                    abs_rank_error[candidate_idx]
                ),
                "fqe_z_overvaluation": float(
                    fqe_z_overvaluation[candidate_idx]
                ),
                "abs_fqe_z_error": float(abs_z_error[candidate_idx]),
                "window_present": bool(
                    cand_total_states[candidate_idx] > 0
                ),
                "window_total_states": int(
                    cand_total_states[candidate_idx]
                ),
                "window_query_states": int(
                    cand_query_states[candidate_idx]
                ),
                "occupancy_mean_knn_radius": float(
                    cand_mean_radius[candidate_idx]
                ),
                "occupancy_median_knn_radius": float(
                    cand_median_radius[candidate_idx]
                ),
                "occupancy_p95_knn_radius": float(
                    cand_p95_radius[candidate_idx]
                ),
                "occupancy_max_knn_radius": float(
                    cand_max_radius[candidate_idx]
                ),
                "occupancy_mean_1nn_distance": float(
                    cand_mean_1nn[candidate_idx]
                ),
                "occupancy_p95_1nn_distance": float(
                    cand_p95_1nn[candidate_idx]
                ),
                "occupancy_ood_fraction": float(
                    cand_ood_fraction[candidate_idx]
                ),
                "occupancy_mean_excess_knn_radius": float(
                    cand_mean_excess[candidate_idx]
                ),
                "occupancy_mean_ratio_to_behavior": float(
                    cand_ratio[candidate_idx]
                ),
                "occupancy_rank": int(
                    occupancy_rank[candidate_idx]
                ),
                "behavior_threshold_knn_radius": float(
                    behavior_threshold
                ),
                "behavior_mean_knn_radius": float(
                    behavior_mean_radius
                ),
                "reference_transitions_window": int(
                    n_window_reference
                ),
                "replay_query_states_window": int(
                    len(replay_query_local_indices)
                ),
            })

    return summary_rows, candidate_rows



def build_support_penalized_scores(
    base_fqe_scores,
    action_divergence,
    support_reference_label,
    support_reference_transitions,
):
    """Combine ordinary FQE value with the explicit behavior-support penalty."""
    if isinstance(base_fqe_scores, LCBFQEScores):
        # Do NOT use the LCB list values here. The new experiment is defined by
        # ordinary FQE value minus explicit behavioral divergence.
        mean_q = np.asarray(base_fqe_scores.mean_q, dtype=np.float64)
        mean_sigma = np.asarray(base_fqe_scores.mean_sigma, dtype=np.float64)
        ensemble_member_values = np.asarray(
            base_fqe_scores.ensemble_member_values, dtype=np.float64
        )
        ensemble_size = int(base_fqe_scores.ensemble_size)
    else:
        mean_q = np.asarray(base_fqe_scores, dtype=np.float64)
        mean_sigma = np.zeros_like(mean_q)
        ensemble_member_values = mean_q.reshape(-1, 1)
        ensemble_size = 1

    action_divergence = np.asarray(action_divergence, dtype=np.float64)
    if mean_q.shape != action_divergence.shape:
        raise RuntimeError(
            "Support-penalized score length mismatch: "
            f"FQE={mean_q.shape}, divergence={action_divergence.shape}."
        )

    support_penalty = FQE_SUPPORT_PENALTY_LAMBDA * action_divergence
    penalized_scores = mean_q - support_penalty

    if not np.all(np.isfinite(penalized_scores)):
        raise RuntimeError(
            "Support-penalized FQE produced a non-finite ranking score."
        )

    result = SupportPenalizedFQEScores(
        scores=penalized_scores,
        mean_q=mean_q,
        action_divergence=action_divergence,
        support_penalty=support_penalty,
        penalty_lambda=FQE_SUPPORT_PENALTY_LAMBDA,
        support_reference_label=support_reference_label,
        support_reference_transitions=support_reference_transitions,
        mean_sigma=mean_sigma,
        ensemble_member_values=ensemble_member_values,
        ensemble_size=ensemble_size,
    )

    # Preserve evaluator-objective metadata for diagnostics/CSV output.
    for attr_name in (
        "fqe_objective",
        "fqe_gamma",
        "finite_horizon_steps",
        "time_conditioned",
        "fqe_n_steps",
        "fqe_target_update_interval",
        "fqe_target_updates",
    ):
        if hasattr(base_fqe_scores, attr_name):
            setattr(result, attr_name, getattr(base_fqe_scores, attr_name))

    return result


def native_batched_fqe(
    model,
    agents,
    dataset,
    native_data=None,
    n_steps=None,
    target_update_interval=None,
):
    """Fit native batched FQE for all candidates on the GPU.

    Everything outside the evaluator remains unchanged. Each candidate policy is
    paired with ``FQE_ENSEMBLE_SIZE`` independently initialized FQE critics,
    trained on the same frozen corrected replay transitions, same candidate
    actions, same transition minibatches, optimizer, and target-update schedule.

    With TIME_CONDITIONED_FINITE_HORIZON_FQE=1, the evaluator objective becomes:
        Q(s_t, a_t, t) = r_t + Q(s_{t+1}, pi(s_{t+1}), t+1)
    with OPE gamma=1 and zero bootstrap at either a true terminal or a replay
    truncation. Only the critic receives the extra normalized time feature t/H;
    candidate PPO policies still act on the original observation.

    Ranking score for candidate pi is computed exactly over the frozen initial
    state reference set:
        mean_s0[
            mean_b Q_b(s0, pi(s0))
            - FQE_LCB_BETA * std_b Q_b(s0, pi(s0))
        ]

    ``std_b`` uses population standard deviation (correction=0 / unbiased=False),
    which is well-defined for B=1. Setting FQE_ENSEMBLE_SIZE=1 and
    FQE_LCB_BETA=0 restores the previous single-critic score.
    """
    if n_steps is None:
        n_steps = FQE_N_STEPS
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("native_batched_fqe n_steps must be > 0.")

    if target_update_interval is None:
        target_update_interval = NATIVE_FQE_TARGET_UPDATE_INTERVAL
    target_update_interval = int(target_update_interval)
    if target_update_interval <= 0:
        raise ValueError(
            "native_batched_fqe target_update_interval must be > 0."
        )

    if len(agents) == 0:
        return LCBFQEScores(
            [], [], [], np.empty((0, FQE_ENSEMBLE_SIZE)),
            FQE_LCB_BETA, FQE_ENSEMBLE_SIZE
        )

    if not isinstance(model.action_space, gym.spaces.Box):
        raise NotImplementedError(
            "Native batched FQE currently supports continuous Box actions only."
        )

    print("--------------------------------------------------------------------------------")
    print(
        "Fitting native batched FQE base estimator for "
        f"{len(agents)} candidates x {FQE_ENSEMBLE_SIZE} critics..."
    )
    print(
        "Native FQE base-estimator runtime config: "
        f"steps={n_steps}, "
        f"batch_size={FQE_BATCH_SIZE}, "
        f"target_update_interval={target_update_interval}, "
        f"hidden_units={NATIVE_FQE_HIDDEN_UNITS}, "
        f"ensemble_size={FQE_ENSEMBLE_SIZE}, "
        f"beta={FQE_LCB_BETA}"
    )

    preparation_start = time.time()
    if native_data is None:
        if TIME_CONDITIONED_FINITE_HORIZON_FQE:
            raise RuntimeError(
                "Time-conditioned finite-horizon FQE requires corrected native "
                "replay data with reconstructed timesteps. Pass native_data "
                "from build_native_fqe_replay_data()."
            )
        # Compatibility fallback for callers outside the active rank-study
        # path. The active native backend passes corrected replay data directly.
        native_data = build_native_fqe_data(dataset)

    time_conditioned_finite_horizon = bool(
        native_data.get(
            "time_conditioned_finite_horizon",
            TIME_CONDITIONED_FINITE_HORIZON_FQE,
        )
    )
    finite_horizon_steps = native_data.get("finite_horizon_steps", None)

    if time_conditioned_finite_horizon:
        if finite_horizon_steps is None:
            raise RuntimeError(
                "Native FQE data is missing finite_horizon_steps."
            )
        finite_horizon_steps = int(finite_horizon_steps)
        if finite_horizon_steps <= 0:
            raise RuntimeError("finite_horizon_steps must be > 0.")
        for required_key in (
            "timesteps",
            "next_timesteps",
            "initial_timesteps",
            "timeouts",
        ):
            if required_key not in native_data:
                raise RuntimeError(
                    "Time-conditioned finite-horizon FQE data is missing "
                    f"{required_key!r}."
                )

        print(
            "Native FQE objective: time-conditioned finite-horizon "
            f"undiscounted return (H={finite_horizon_steps}, OPE gamma=1.0)."
        )
    else:
        print(
            "Native FQE objective: original discounted continuing objective "
            f"(gamma={float(model.gamma):.8f})."
        )

    obs_cpu = native_data["observations"]
    action_cpu = native_data["actions"]
    reward_cpu = native_data["rewards"]
    next_obs_cpu = native_data["next_observations"]
    terminal_cpu = native_data["terminals"]
    timeout_cpu = native_data.get(
        "timeouts", np.zeros_like(terminal_cpu, dtype=np.float32)
    )
    interval_cpu = native_data["intervals"]
    initial_obs_cpu = native_data["initial_observations"]

    if time_conditioned_finite_horizon:
        timestep_cpu = native_data["timesteps"]
        next_timestep_cpu = native_data["next_timesteps"]
        initial_timestep_cpu = native_data["initial_timesteps"]
    else:
        timestep_cpu = None
        next_timestep_cpu = None
        initial_timestep_cpu = None

    raw_observation_dim = int(obs_cpu.shape[1])
    observation_dim = (
        raw_observation_dim + 1
        if time_conditioned_finite_horizon
        else raw_observation_dim
    )
    action_dim = int(action_cpu.shape[1])
    transition_count = int(obs_cpu.shape[0])
    n_candidates = len(agents)
    n_ensemble = FQE_ENSEMBLE_SIZE

    # Transfer the frozen dataset to the GPU once.
    observations = torch.as_tensor(obs_cpu, dtype=torch.float32, device=device)
    dataset_actions = torch.as_tensor(
        action_cpu, dtype=torch.float32, device=device
    )
    rewards = torch.as_tensor(reward_cpu, dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(
        next_obs_cpu, dtype=torch.float32, device=device
    )
    terminals = torch.as_tensor(
        terminal_cpu, dtype=torch.float32, device=device
    )
    timeouts = torch.as_tensor(
        timeout_cpu, dtype=torch.float32, device=device
    )
    intervals = torch.as_tensor(
        interval_cpu, dtype=torch.float32, device=device
    )
    initial_observations = torch.as_tensor(
        initial_obs_cpu, dtype=torch.float32, device=device
    )

    if time_conditioned_finite_horizon:
        timesteps = torch.as_tensor(
            timestep_cpu, dtype=torch.float32, device=device
        )
        next_timesteps = torch.as_tensor(
            next_timestep_cpu, dtype=torch.float32, device=device
        )
        initial_timesteps = torch.as_tensor(
            initial_timestep_cpu, dtype=torch.float32, device=device
        )

        critic_observations = _append_normalized_fqe_time(
            observations, timesteps, finite_horizon_steps
        )
        critic_next_observations = _append_normalized_fqe_time(
            next_observations, next_timesteps, finite_horizon_steps
        )
        critic_initial_observations = _append_normalized_fqe_time(
            initial_observations,
            initial_timesteps,
            finite_horizon_steps,
        )

        # The online evaluator stops at either environment termination or
        # truncation, so finite-horizon FQE must stop bootstrapping at both.
        fqe_terminals = torch.maximum(terminals, timeouts)
    else:
        critic_observations = observations
        critic_next_observations = next_observations
        critic_initial_observations = initial_observations
        fqe_terminals = terminals

    # Precompute pi_j(s') and pi_j(s0) once per fixed candidate policy. All B
    # ensemble critics for candidate j evaluate the exact same target policy.
    cached_next_actions = []
    cached_initial_actions = []

    for candidate_index, agent in enumerate(agents):
        model.policy.load_state_dict(agent)
        model.policy.to(device)

        next_actions = _policy_actions_current_model(
            model, next_observations
        )
        init_actions = _policy_actions_current_model(
            model, initial_observations
        )

        cached_next_actions.append(next_actions)
        cached_initial_actions.append(init_actions)

    cached_next_actions = torch.stack(cached_next_actions, dim=0)
    cached_initial_actions = torch.stack(cached_initial_actions, dim=0)

    # Preserve the previous d3rlpy/native alignment: two preliminary NumPy
    # transition draws precede the shared training minibatch schedule.
    _ = dataset.sample_transition()
    _ = dataset.sample_transition()

    # Candidate x ensemble critic bank. Ensemble initializations differ across
    # member b but the same initialization set is reused across candidates.
    critic = BatchedFQECritic(
        n_candidates=n_candidates,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_units=NATIVE_FQE_HIDDEN_UNITS,
        n_ensemble=n_ensemble,
        compute_device=device,
    )
    target_critic = copy.deepcopy(critic)
    target_critic.requires_grad_(False)

    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )

    # Every candidate and every ensemble member sees the same sampled replay
    # transitions at each gradient step. Thus disagreement is induced by critic
    # initialization rather than by accidental differences in data exposure.
    sampled_indices_np = np.random.randint(
        0,
        transition_count,
        size=(n_steps, FQE_BATCH_SIZE),
    )
    sampled_indices = torch.as_tensor(
        sampled_indices_np, dtype=torch.long, device=device
    )
    del sampled_indices_np

    if time_conditioned_finite_horizon:
        # Match the original online selector's undiscounted finite-episode sum.
        fqe_gamma = 1.0
    else:
        fqe_gamma = float(model.gamma)

    gamma_tensor = torch.as_tensor(
        fqe_gamma, dtype=torch.float32, device=device
    )
    discounts = torch.pow(gamma_tensor, intervals)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.time() - preparation_start
    fit_start = time.time()

    last_losses = None
    for grad_step in range(n_steps):
        batch_index = sampled_indices[grad_step]

        obs_batch = critic_observations[batch_index]
        action_batch = dataset_actions[batch_index]
        reward_batch = rewards[batch_index]
        next_obs_batch = critic_next_observations[batch_index]
        terminal_batch = fqe_terminals[batch_index]
        # [candidate, batch, action_dim]; BatchedFQECritic broadcasts this over
        # each candidate's ensemble dimension.
        next_action_batch = cached_next_actions[:, batch_index, :]

        with torch.no_grad():
            # [candidate, ensemble, batch, 1]
            target_q = target_critic(next_obs_batch, next_action_batch)
            discount = discounts[batch_index].unsqueeze(0).unsqueeze(0)
            bellman_target = (
                reward_batch.unsqueeze(0).unsqueeze(0)
                + discount
                * target_q
                * (1.0 - terminal_batch.unsqueeze(0).unsqueeze(0))
            )

        predicted_q = critic(obs_batch, action_batch)

        # Each ensemble member is an independent FQE critic. Sum the member-wise
        # mean losses so every critic receives the same gradient magnitude it
        # would receive if optimized separately.
        squared_error = (predicted_q - bellman_target).pow(2)
        loss_per_critic = squared_error.mean(dim=(2, 3))
        loss = loss_per_critic.sum()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Native batched LCB-FQE produced a non-finite loss at "
                f"gradient step {grad_step}."
            )

        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.step()

        # Match d3rlpy FQEImpl: target update happens after optimizer.step(),
        # including grad_step == 0.
        if grad_step % target_update_interval == 0:
            target_critic.load_state_dict(critic.state_dict())

        last_losses = loss_per_critic.detach()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.time() - fit_start

    with torch.no_grad():
        # [candidate, ensemble, n_initial_states]
        initial_q = critic(
            critic_initial_observations,
            cached_initial_actions,
        ).squeeze(-1)

        if not torch.all(torch.isfinite(initial_q)):
            raise RuntimeError(
                "Native batched LCB-FQE produced a non-finite initial-state Q."
            )

        # Implement the requested objective literally: first calculate ensemble
        # mean/std at EACH s0, then form the lower confidence bound at that s0,
        # then average over the frozen initial-state distribution.
        state_mean_q = initial_q.mean(dim=1)
        state_sigma_q = initial_q.std(dim=1, unbiased=False)
        state_lcb_q = state_mean_q - FQE_LCB_BETA * state_sigma_q

        lcb_scores = state_lcb_q.mean(dim=1)
        mean_q_scores = state_mean_q.mean(dim=1)
        mean_sigma_scores = state_sigma_q.mean(dim=1)

        # Analysis-only diagnostic: each member's ordinary FQE initial-state
        # value. This is not the quantity used to compute the per-state LCB.
        ensemble_member_values = initial_q.mean(dim=2)

    if not torch.all(torch.isfinite(lcb_scores)):
        raise RuntimeError(
            "Native batched LCB-FQE produced a non-finite LCB score."
        )

    scores_np = (
        lcb_scores.detach().cpu().numpy().astype(np.float64)
    )
    mean_q_np = (
        mean_q_scores.detach().cpu().numpy().astype(np.float64)
    )
    mean_sigma_np = (
        mean_sigma_scores.detach().cpu().numpy().astype(np.float64)
    )
    ensemble_member_values_np = (
        ensemble_member_values.detach().cpu().numpy().astype(np.float64)
    )

    scores = LCBFQEScores(
        scores=scores_np.tolist(),
        mean_q=mean_q_np,
        mean_sigma=mean_sigma_np,
        ensemble_member_values=ensemble_member_values_np,
        beta=FQE_LCB_BETA,
        ensemble_size=n_ensemble,
    )
    scores.fqe_gamma = float(fqe_gamma)
    scores.fqe_n_steps = int(n_steps)
    scores.fqe_target_update_interval = int(target_update_interval)
    scores.fqe_target_updates = int(
        ((n_steps - 1) // target_update_interval) + 1
    )
    scores.time_conditioned = bool(time_conditioned_finite_horizon)
    scores.finite_horizon_steps = (
        int(finite_horizon_steps)
        if time_conditioned_finite_horizon
        else None
    )
    scores.fqe_objective = (
        "time_conditioned_finite_horizon_undiscounted"
        if time_conditioned_finite_horizon
        else "original_discounted_continuing"
    )
    if last_losses is not None:
        scores.final_loss_per_candidate = (
            last_losses.mean(dim=1).detach().cpu().numpy().astype(np.float64)
        )
        scores.mean_final_loss = float(np.mean(scores.final_loss_per_candidate))
    else:
        scores.final_loss_per_candidate = np.full(
            n_candidates, np.nan, dtype=np.float64
        )
        scores.mean_final_loss = float("nan")

    print(
        f"Native batched FQE preparation time: {preparation_seconds:.3f} s"
    )
    print(
        "Native batched FQE fitting time for all "
        f"{n_candidates * n_ensemble} critics: {fit_seconds:.3f} s"
    )
    if last_losses is not None:
        print(
            "Final mean FQE loss per candidate (averaged over ensemble): "
            + np.array2string(
                last_losses.mean(dim=1).detach().cpu().numpy(),
                precision=4,
                separator=", ",
                max_line_width=160,
            )
        )
    print(
        "Ensemble mean initial-state Q: "
        + np.array2string(
            mean_q_np,
            precision=6,
            separator=", ",
            max_line_width=160,
        )
    )
    print(
        "Mean per-state ensemble sigma: "
        + np.array2string(
            mean_sigma_np,
            precision=6,
            separator=", ",
            max_line_width=160,
        )
    )
    if FQE_ENSEMBLE_SIZE > 1 or FQE_LCB_BETA > 0.0:
        print(
            f"Legacy LCB diagnostic (NOT used by support-penalized ranking; "
            f"beta={FQE_LCB_BETA:g}): "
            + np.array2string(
                scores_np,
                precision=6,
                separator=", ",
                max_line_width=160,
            )
        )

    return scores

def native_batched_fqe_preserving_rng(
    model,
    agents,
    dataset,
    native_data=None,
    n_steps=None,
    target_update_interval=None,
):
    """Run native batched FQE without changing the online oracle RNG trajectory."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    try:
        return native_batched_fqe(
            model,
            agents,
            dataset,
            native_data=native_data,
            n_steps=n_steps,
            target_update_interval=target_update_interval,
        )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def sequential_d3rlpy_fqe_scores_preserving_rng(
    model, agents, dataset, dir_name, iteration
):
    """Validation/fallback path matching the previous sequential d3rlpy study."""
    scores = []
    for candidate_index, agent in enumerate(agents):
        model.policy.load_state_dict(agent)
        model.policy.to(device)
        estimate = d3rl_evaluation_preserving_rng(
            model,
            (
                f"{'-'.join(dir_name.split('/'))}"
                f"-rank-iter{iteration}-agent{candidate_index}"
            ),
            dataset,
        )
        if estimate is None:
            raise RuntimeError(
                f"d3rlpy FQE failed for iteration {iteration}, "
                f"agent {candidate_index}."
            )
        scores.append(float(np.asarray(estimate).reshape(-1)[0]))
    return scores



# -------------------------------------------------------------------------------------------------
# FQE replay-collection semantics
# -------------------------------------------------------------------------------------------------
# Version 2 means the replay buffer stores the transition that was actually experienced by the
# environment:
#   observation      = state before env.step
#   action           = action actually sent to env.step
#   reward           = raw reward returned by env.step (before PPO TimeLimit bootstrap correction)
#   next_observation = true terminal/truncated observation when an episode ends
#   done/timeout     = original environment boundary flags
#
# PPO's rollout buffer and PPO training logic are intentionally left unchanged.
FQE_REPLAY_SEMANTICS_VERSION = 2


def _copy_vec_observation(observation):
    """Copy a VecEnv observation without changing its structure."""
    if isinstance(observation, dict):
        return {
            key: np.array(value, copy=True)
            for key, value in observation.items()
        }
    return np.array(observation, copy=True)


def _set_vec_observation_at(observation, env_index, value):
    """Replace one environment slot in a copied VecEnv observation."""
    if isinstance(observation, dict):
        if not isinstance(value, dict):
            raise TypeError(
                "terminal_observation structure does not match Dict observation."
            )
        for key in observation:
            observation[key][env_index] = value[key]
    else:
        observation[env_index] = value


def install_fqe_replay_semantics_patch(model):
    """Patch only ReplayBuffer.add so FQE receives scientifically correct transitions.

    The custom PPO collector used by this project performs PPO-specific TimeLimit handling:
      1) the environment is stepped with clipped/unscaled actions,
      2) TimeLimit rewards are augmented with gamma * V(terminal_observation),
      3) replay_buffer.add(...) is called with PPO-facing actions/rewards/new_obs.

    Those PPO-facing values are correct for PPO training, but they are not the raw transition
    tuple required by offline policy evaluation. This patch intercepts only the replay-buffer
    write and reconstructs the raw environment transition. It does NOT modify:
      * env.step(...)
      * rewards used by PPO's rollout buffer
      * actions used by PPO's rollout buffer
      * PPO advantages/returns
      * callbacks, timesteps, or _last_obs
      * ESA candidate generation or online candidate selection
    """
    rb = model.replay_buffer

    if getattr(rb, "_fqe_replay_semantics_patch_installed", False):
        return

    original_add = rb.add
    action_space = model.action_space

    # Diagnostic counters only; they never participate in training or selection.
    stats = {
        "transitions_written": 0,
        "actions_changed_for_replay": 0,
        "timeout_rewards_restored": 0,
        "terminal_next_obs_restored": 0,
    }

    def corrected_replay_add(obs, next_obs, action, reward, done, infos):
        # Work exclusively on copies so PPO-facing variables owned by collect_rollouts()
        # remain bit-for-bit untouched.
        replay_obs = _copy_vec_observation(obs)
        replay_next_obs = _copy_vec_observation(next_obs)
        replay_actions = np.array(action, copy=True)
        replay_rewards = np.array(reward, copy=True)
        replay_dones = np.array(done, copy=True)

        # ------------------------------------------------------------------
        # 1) Store the action that was ACTUALLY sent to env.step().
        # ------------------------------------------------------------------
        if isinstance(action_space, gym.spaces.Box):
            original_replay_actions = replay_actions.copy()

            if model.policy.squash_output:
                # Matches SB3 collect_rollouts(): env.step(policy.unscale_action(actions))
                replay_actions = model.policy.unscale_action(replay_actions)
            else:
                # Matches SB3 collect_rollouts(): env.step(np.clip(actions, low, high))
                replay_actions = np.clip(
                    replay_actions,
                    action_space.low,
                    action_space.high,
                )

            stats["actions_changed_for_replay"] += int(
                np.count_nonzero(
                    np.any(
                        np.not_equal(
                            replay_actions,
                            original_replay_actions,
                        ),
                        axis=-1,
                    )
                )
            )

        # ------------------------------------------------------------------
        # 2) Restore true next observations at episode boundaries.
        #
        # VecEnv auto-resets environments, so `next_obs[idx]` is normally the
        # RESET observation when done=True. The environment transition itself
        # ended at infos[idx]["terminal_observation"].
        # ------------------------------------------------------------------
        done_flags = np.asarray(replay_dones).reshape(-1)
        for idx, done_flag in enumerate(done_flags):
            if not bool(done_flag):
                continue

            info = infos[idx]
            terminal_observation = info.get("terminal_observation")
            if terminal_observation is not None:
                _set_vec_observation_at(
                    replay_next_obs,
                    idx,
                    terminal_observation,
                )
                stats["terminal_next_obs_restored"] += 1

            # --------------------------------------------------------------
            # 3) Restore the RAW environment reward at TimeLimit truncation.
            #
            # The custom PPO collector adds:
            #     gamma * V(terminal_observation)
            # to PPO's reward before replay_buffer.add(...).
            #
            # Recompute exactly that bootstrap term and subtract it from the
            # replay-only copy. PPO continues using its corrected reward.
            # --------------------------------------------------------------
            if (
                terminal_observation is not None
                and info.get("TimeLimit.truncated", False)
            ):
                terminal_obs_tensor = model.policy.obs_to_tensor(
                    terminal_observation
                )[0]

                with torch.no_grad():
                    terminal_value = model.policy.predict_values(
                        terminal_obs_tensor
                    )[0]

                bootstrap_value = (
                    float(model.gamma)
                    * float(terminal_value.detach().cpu().item())
                )

                # rewards is [n_envs] in the active collector. Handle a
                # possible trailing singleton dimension defensively.
                if replay_rewards.ndim == 1:
                    replay_rewards[idx] -= bootstrap_value
                else:
                    replay_rewards[idx, ...] -= bootstrap_value

                stats["timeout_rewards_restored"] += 1

        stats["transitions_written"] += int(len(done_flags))

        # Preserve the original ReplayBuffer.add implementation and therefore
        # its ring-buffer position/full logic and timeout extraction from infos.
        result = original_add(
            replay_obs,
            replay_next_obs,
            replay_actions,
            replay_rewards,
            replay_dones,
            infos,
        )

        # Diagnostic/FQE metadata only. One ReplayBuffer.add call writes one
        # ring position (containing one transition per VecEnv slot). This never
        # feeds back into PPO training or modifies stored transitions.
        if hasattr(rb, "_fqe_resume_positions_written"):
            rb._fqe_resume_positions_written += 1

        return result

    rb.add = corrected_replay_add
    rb._fqe_replay_semantics_patch_installed = True
    rb._fqe_replay_semantics_version = FQE_REPLAY_SEMANTICS_VERSION
    rb._fqe_replay_semantics_stats = stats

    print(
        "Installed FQE replay-semantics patch: "
        "executed actions + raw rewards + true terminal next observations."
    )


def print_fqe_replay_semantics_stats(model):
    """Print replay-only correction counts for debugging/validation."""
    rb = model.replay_buffer
    stats = getattr(rb, "_fqe_replay_semantics_stats", None)
    if stats is None:
        print("FQE replay-semantics patch statistics unavailable.")
        return

    print(
        "FQE replay semantics stats: "
        f"written={stats['transitions_written']}, "
        f"action_corrections={stats['actions_changed_for_replay']}, "
        f"timeout_reward_corrections={stats['timeout_rewards_restored']}, "
        f"terminal_next_obs_corrections={stats['terminal_next_obs_restored']}"
    )


def save_replay_buffer_npz(model, path):
    """Save the complete SB3 ReplayBuffer state needed for exact restoration."""
    rb = model.replay_buffer

    # Do not allow an unpatched/legacy replay buffer to be mislabeled as
    # corrected FQE data merely because this newer save helper is being used.
    semantics_version = getattr(
        rb, "_fqe_replay_semantics_version", None
    )
    if semantics_version != FQE_REPLAY_SEMANTICS_VERSION:
        raise RuntimeError(
            "Refusing to save replay buffer as corrected FQE data because "
            f"its semantics version is {semantics_version!r}, expected "
            f"{FQE_REPLAY_SEMANTICS_VERSION}. Install the replay-semantics "
            "patch before collecting this buffer."
        )

    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "observations": np.asarray(rb.observations),
        "actions": np.asarray(rb.actions),
        "rewards": np.asarray(rb.rewards),
        "dones": np.asarray(rb.dones),
        "timeouts": np.asarray(rb.timeouts),
        "pos": np.asarray(rb.pos, dtype=np.int64),
        "full": np.asarray(rb.full, dtype=np.bool_),
        "buffer_size": np.asarray(rb.buffer_size, dtype=np.int64),
        "n_envs": np.asarray(rb.n_envs, dtype=np.int64),
        "optimize_memory_usage": np.asarray(
            getattr(rb, "optimize_memory_usage", False), dtype=np.bool_
        ),
        "handle_timeout_termination": np.asarray(
            getattr(rb, "handle_timeout_termination", True), dtype=np.bool_
        ),
        "fqe_replay_semantics_version": np.asarray(
            FQE_REPLAY_SEMANTICS_VERSION, dtype=np.int64
        ),
    }

    if hasattr(rb, "next_observations"):
        payload["next_observations"] = np.asarray(rb.next_observations)

    np.savez(path, **payload)

    print("Replay buffer saved")
    print("  path:", path)
    print("  pos:", rb.pos)
    print("  full:", rb.full)
    print("  size:", rb.size())
    print("  dones:", int(np.sum(rb.dones)))
    print("  timeouts:", int(np.sum(rb.timeouts)))


def load_replay_buffer_npz(model, path):
    """Restore a replay buffer saved by save_replay_buffer_npz exactly.

    Old flattened files are intentionally rejected because they are missing
    timeout and circular-buffer metadata needed for reliable FQE.
    """
    rb = model.replay_buffer
    path = str(path)

    required = {
        "observations",
        "actions",
        "rewards",
        "dones",
        "timeouts",
        "pos",
        "full",
        "buffer_size",
        "n_envs",
        "fqe_replay_semantics_version",
    }

    with np.load(path, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise RuntimeError(
                "Replay buffer file is in the old/incomplete format and cannot be "
                "restored exactly for FQE. Missing fields: "
                f"{missing}. Regenerate the initial replay buffer with the "
                "corrected replay collector and save_replay_buffer_npz(). "
                "Existing buffers created before the replay-semantics fix "
                "cannot be repaired reliably after the fact."
            )

        saved_buffer_size = int(np.asarray(saved["buffer_size"]).item())
        saved_n_envs = int(np.asarray(saved["n_envs"]).item())
        saved_semantics_version = int(
            np.asarray(saved["fqe_replay_semantics_version"]).item()
        )

        if saved_semantics_version != FQE_REPLAY_SEMANTICS_VERSION:
            raise RuntimeError(
                "Replay-buffer FQE semantics mismatch: "
                f"file={saved_semantics_version}, "
                f"required={FQE_REPLAY_SEMANTICS_VERSION}. "
                "Regenerate the initial replay buffer with the corrected "
                "replay-collection semantics before running FQE."
            )

        if saved_buffer_size != rb.buffer_size:
            raise RuntimeError(
                "Replay-buffer size mismatch: "
                f"file={saved_buffer_size}, model={rb.buffer_size}"
            )
        if saved_n_envs != rb.n_envs:
            raise RuntimeError(
                "Replay-buffer n_envs mismatch: "
                f"file={saved_n_envs}, model={rb.n_envs}"
            )

        def restore_array(name, target):
            source = np.asarray(saved[name])
            if source.shape != target.shape:
                raise RuntimeError(
                    f"Replay-buffer {name} shape mismatch: "
                    f"file={source.shape}, model={target.shape}"
                )
            target[...] = source

        # Preserve SB3's original [buffer_size, n_envs, ...] array layout.
        restore_array("observations", rb.observations)
        restore_array("actions", rb.actions)
        restore_array("rewards", rb.rewards)
        restore_array("dones", rb.dones)
        restore_array("timeouts", rb.timeouts)

        saved_optimize = bool(
            np.asarray(saved["optimize_memory_usage"]).item()
        ) if "optimize_memory_usage" in saved.files else False
        current_optimize = bool(getattr(rb, "optimize_memory_usage", False))
        if saved_optimize != current_optimize:
            raise RuntimeError(
                "Replay-buffer optimize_memory_usage mismatch: "
                f"file={saved_optimize}, model={current_optimize}"
            )

        if hasattr(rb, "next_observations"):
            if "next_observations" not in saved.files:
                raise RuntimeError(
                    "Replay buffer file is missing next_observations."
                )
            restore_array("next_observations", rb.next_observations)

        if "handle_timeout_termination" in saved.files:
            rb.handle_timeout_termination = bool(
                np.asarray(saved["handle_timeout_termination"]).item()
            )

        rb.pos = int(np.asarray(saved["pos"]).item())
        rb.full = bool(np.asarray(saved["full"]).item())

        # FQE-only resume-seam metadata. The environment state itself is not
        # serialized with this replay buffer. After loading, the experiment
        # explicitly resets the environment before model.learn() resumes, so
        # the first newly written replay transition begins a NEW episode even
        # if the newest saved transition belonged to an unfinished episode.
        #
        # Record the next replay position now; corrected_replay_add() will count
        # how many new ring positions have been written after the load. Native
        # FQE uses this metadata only to avoid concatenating an old partial
        # episode with a post-reset episode. No replay contents or PPO variables
        # are changed.
        rb._fqe_resume_seam_pos = int(rb.pos)
        rb._fqe_resume_positions_written = 0

        # The file version has already been validated above. Restore it onto
        # the live replay object instead of relying only on patch-install order.
        rb._fqe_replay_semantics_version = saved_semantics_version

    if not (0 <= rb.pos < rb.buffer_size):
        raise RuntimeError(
            f"Invalid restored replay-buffer position {rb.pos} "
            f"for buffer_size={rb.buffer_size}."
        )

    print("Replay buffer loaded")
    print("  path:", path)
    print("  pos:", rb.pos)
    print("  full:", rb.full)
    print("  size:", rb.size())
    print("  observations:", rb.observations.shape)
    print("  actions:", rb.actions.shape)
    print("  rewards:", rb.rewards.shape)
    print("  dones:", int(np.sum(rb.dones)))
    print("  timeouts:", int(np.sum(rb.timeouts)))
    print("  FQE replay semantics version:", FQE_REPLAY_SEMANTICS_VERSION)


def build_fqe_dataset(model):
    """Build a frozen d3rlpy dataset from valid replay-buffer episodes.

    The replay buffer is circular. Chronological order is restored with pos/full.
    Real TimeLimit truncations come from ReplayBuffer.timeouts. No artificial
    terminal or timeout flags are created.

    For a full circular buffer, the oldest retained sample may begin mid-episode.
    The leading partial episode is discarded. The newest incomplete episode is
    also discarded. Only complete trajectories are given to d3rlpy.
    """
    rb = model.replay_buffer

    if rb.size() == 0:
        raise RuntimeError("Cannot build FQE dataset from an empty PPO replay buffer.")

    print("Replay buffer:")
    print("  pos:", rb.pos)
    print("  full:", rb.full)
    print("  size:", rb.size())
    print("  dones:", int(np.sum(rb.dones)))
    print("  timeouts:", int(np.sum(rb.timeouts)))

    if rb.full:
        time_indices = np.concatenate(
            (
                np.arange(rb.pos, rb.buffer_size, dtype=np.int64),
                np.arange(0, rb.pos, dtype=np.int64),
            )
        )
    else:
        time_indices = np.arange(0, rb.pos, dtype=np.int64)

    # Keep the auxiliary d3rlpy dataset consistent with the corrected native
    # finite-horizon replay view. A replay buffer loaded from disk is followed
    # by an explicit environment reset in this experiment, so the unfinished
    # saved trajectory immediately before the first post-load write cannot be
    # concatenated with the post-reset trajectory. This correction is enabled
    # only for the new finite-horizon study so disabling that study preserves
    # the legacy dataset path exactly.
    resume_seam_local_index = None
    resume_seam_pos = getattr(rb, "_fqe_resume_seam_pos", None)
    resume_positions_written = int(
        getattr(rb, "_fqe_resume_positions_written", 0)
    )
    if (
        TIME_CONDITIONED_FINITE_HORIZON_FQE
        and resume_seam_pos is not None
        and resume_positions_written > 0
        and resume_positions_written < rb.buffer_size
    ):
        seam_matches = np.flatnonzero(
            time_indices == int(resume_seam_pos)
        )
        if len(seam_matches) > 1:
            raise RuntimeError(
                "Internal FQE dataset error: resume seam appeared more than "
                "once in chronological replay indices."
            )
        if len(seam_matches) == 1:
            resume_seam_local_index = int(seam_matches[0])

    obs_raw = np.asarray(rb.observations)[time_indices]
    actions_raw = np.asarray(rb.actions)[time_indices]
    rewards_raw = np.asarray(rb.rewards)[time_indices]
    dones_raw = np.asarray(rb.dones)[time_indices]
    timeouts_raw = np.asarray(rb.timeouts)[time_indices]

    observations_parts = []
    actions_parts = []
    rewards_parts = []
    terminals_parts = []
    timeouts_parts = []
    resume_gap_discarded_transitions = 0

    for env_idx in range(rb.n_envs):
        obs_env = np.asarray(obs_raw[:, env_idx]).copy()
        actions_env = np.asarray(actions_raw[:, env_idx]).copy()
        rewards_env = np.asarray(rewards_raw[:, env_idx]).reshape(-1, 1).copy()
        dones_env = np.asarray(dones_raw[:, env_idx]).reshape(-1, 1).astype(
            np.float32, copy=True
        )
        timeouts_env = np.asarray(timeouts_raw[:, env_idx]).reshape(-1, 1).astype(
            np.float32, copy=True
        )

        # Match SB3 ReplayBuffer sampling semantics: TimeLimit truncation is
        # an episode boundary but not an environmental terminal.
        terminals_env = dones_env * (1.0 - timeouts_env)

        begins_at_known_resume_start = False
        if resume_seam_local_index is not None:
            seam = int(resume_seam_local_index)
            if not (0 <= seam < len(obs_env)):
                raise RuntimeError(
                    "Internal FQE dataset error: resume seam index is outside "
                    "the selected replay view."
                )

            raw_boundary_flags = (
                (terminals_env[:, 0] > 0.5)
                | (timeouts_env[:, 0] > 0.5)
            )
            boundaries_before_seam = np.flatnonzero(
                raw_boundary_flags[:seam]
            )
            prefix_end = (
                int(boundaries_before_seam[-1] + 1)
                if len(boundaries_before_seam) > 0
                else 0
            )
            discarded_here = int(seam - prefix_end)
            if discarded_here < 0:
                raise RuntimeError(
                    "Internal FQE dataset error: negative resume-gap length."
                )

            if discarded_here > 0 or seam == 0:
                def _join_dataset_across_resume(arr):
                    return np.concatenate(
                        (arr[:prefix_end], arr[seam:]), axis=0
                    )

                obs_env = _join_dataset_across_resume(obs_env)
                actions_env = _join_dataset_across_resume(actions_env)
                rewards_env = _join_dataset_across_resume(rewards_env)
                dones_env = _join_dataset_across_resume(dones_env)
                timeouts_env = _join_dataset_across_resume(timeouts_env)
                terminals_env = _join_dataset_across_resume(terminals_env)
                resume_gap_discarded_transitions += discarded_here
                begins_at_known_resume_start = (prefix_end == 0)

        boundaries = np.flatnonzero(
            (terminals_env[:, 0] > 0.5) | (timeouts_env[:, 0] > 0.5)
        )
        if len(boundaries) == 0:
            raise RuntimeError(
                f"No complete episode boundary was found in replay-buffer env {env_idx}. "
                "FQE requires stored terminal/timeout metadata."
            )

        # If full, the first retained transition can be in the middle of an
        # overwritten episode, so begin immediately after the first boundary.
        # The one exception is a replay view that now begins exactly at the
        # known post-load environment reset.
        if begins_at_known_resume_start:
            start = 0
        else:
            start = int(boundaries[0] + 1) if rb.full else 0

        # Stop at the last real boundary so no incomplete newest trajectory is
        # presented to d3rlpy as a complete episode.
        end = int(boundaries[-1] + 1)

        if start >= end:
            raise RuntimeError(
                f"Replay-buffer env {env_idx} contains no complete episode after "
                "removing partial circular-buffer trajectories."
            )

        observations_parts.append(obs_env[start:end])
        actions_parts.append(actions_env[start:end])
        rewards_parts.append(rewards_env[start:end])
        terminals_parts.append(terminals_env[start:end])
        timeouts_parts.append(timeouts_env[start:end])

    observations = np.concatenate(observations_parts, axis=0)
    actions = np.concatenate(actions_parts, axis=0)
    rewards = np.concatenate(rewards_parts, axis=0).astype(np.float32, copy=False)
    # d3rlpy expects terminal/timeout flags as one value per transition.
    # Passing both as 1-D also avoids a broadcasting bug in EpisodeGenerator,
    # which flattens terminals internally but does not flatten timeouts.
    terminals = np.concatenate(
        terminals_parts, axis=0
    ).reshape(-1).astype(
        np.float32, copy=False
    )

    timeouts = np.concatenate(
        timeouts_parts, axis=0
    ).reshape(-1).astype(
        np.float32, copy=False
    )

    # Verify that our reconstructed dataset is semantically valid
    if np.any(
        np.logical_and(
            terminals > 0.5,
            timeouts > 0.5
        )
    ):
        raise RuntimeError(
            "Internal FQE dataset error: at least one transition "
            "is marked as both terminal and timeout."
        )

    n_episodes = int(
        np.sum(
            (terminals > 0.5)
            | (timeouts > 0.5)
        )
    )

    print(
        "FQE dataset: "
        f"{len(terminals)} complete-episode transitions, "
        f"{int(np.sum(terminals))} true terminals, "
        f"{int(np.sum(timeouts))} timeouts, "
        f"{n_episodes} episodes"
    )
    if resume_gap_discarded_transitions > 0:
        print(
            "  FQE dataset replay-resume seam: discarded "
            f"{resume_gap_discarded_transitions} transition(s) from the "
            "unfinished saved trajectory before the post-load reset."
        )

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )




def compute_replay_coverage_rank_metrics(
    online_scores,
    fqe_scores,
    iteration,
    coverage_label,
    native_data,
):
    """Compute direct-ranking and hybrid-shortlist metrics for one replay window.

    Hybrid metrics answer the intended deployment question: after FQE ranks all
    candidates, if only its top-k are evaluated online, does that shortlist
    contain the true oracle-best candidate and what regret remains after taking
    the best online return within that shortlist?
    """
    is_support_penalized = isinstance(
        fqe_scores, SupportPenalizedFQEScores
    )
    fqe_ensemble_size = int(
        getattr(fqe_scores, "ensemble_size", 1)
    )
    support_lambda = float(
        getattr(fqe_scores, "penalty_lambda", 0.0)
    )
    support_reference_label = str(
        getattr(fqe_scores, "support_reference_label", "none")
    )
    support_reference_transitions = int(
        getattr(fqe_scores, "support_reference_transitions", 0)
    )
    fqe_objective = str(
        getattr(fqe_scores, "fqe_objective", "unknown")
    )
    fqe_gamma = float(
        getattr(fqe_scores, "fqe_gamma", np.nan)
    )
    finite_horizon_steps = getattr(
        fqe_scores, "finite_horizon_steps", None
    )
    time_conditioned = bool(
        getattr(fqe_scores, "time_conditioned", False)
    )
    online_scores = np.asarray(online_scores, dtype=np.float64)
    fqe_scores = np.asarray(fqe_scores, dtype=np.float64)

    if len(online_scores) != len(fqe_scores):
        raise RuntimeError(
            "Replay-coverage rank length mismatch: "
            f"{len(online_scores)} online vs {len(fqe_scores)} FQE."
        )
    if len(online_scores) == 0:
        raise RuntimeError("Replay-coverage metrics received zero candidates.")

    rank_df = pd.DataFrame({"fqe": fqe_scores, "online": online_scores})
    pearson = float(rank_df.corr(method="pearson").loc["fqe", "online"])
    spearman = float(rank_df.corr(method="spearman").loc["fqe", "online"])
    kendall = float(rank_df.corr(method="kendall").loc["fqe", "online"])

    n_candidates = len(online_scores)
    oracle_idx = int(np.argmax(online_scores))
    fqe_idx = int(np.argmax(fqe_scores))
    oracle_return = float(online_scores[oracle_idx])
    selected_return = float(online_scores[fqe_idx])
    online_order = np.argsort(online_scores)[::-1]
    fqe_order = np.argsort(fqe_scores)[::-1]

    # There can be more than one oracle-optimal candidate when online returns
    # tie exactly. Keep oracle_idx for backward-compatible reporting, but make
    # hybrid recall tie-safe: retaining ANY candidate with the oracle return is
    # sufficient for a hybrid shortlist to achieve zero oracle regret.
    oracle_best_mask = online_scores == oracle_return

    metrics = {
        "iteration": int(iteration),
        "coverage": str(coverage_label),
        "requested_max_transitions": (
            -1 if native_data["requested_max_transitions"] is None
            else int(native_data["requested_max_transitions"])
        ),
        "selected_raw_transitions": int(native_data["selected_raw_transitions"]),
        "actual_transitions": int(native_data["actual_transitions"]),
        "n_episodes": int(native_data["n_episodes"]),
        "training_initial_states": int(
            native_data.get(
                "training_initial_states",
                len(native_data["initial_observations"]),
            )
        ),
        "score_initial_states": int(
            native_data.get(
                "score_initial_states",
                len(native_data["initial_observations"]),
            )
        ),
        "fqe_score_type": (
            "support_penalized" if is_support_penalized else "mean"
        ),
        "fqe_ensemble_size": fqe_ensemble_size,
        "support_penalty_lambda": support_lambda,
        "support_reference": support_reference_label,
        "support_reference_transitions": support_reference_transitions,
        "fqe_objective": fqe_objective,
        "fqe_gamma": fqe_gamma,
        "finite_horizon_steps": (
            -1 if finite_horizon_steps is None
            else int(finite_horizon_steps)
        ),
        "time_conditioned": time_conditioned,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "oracle_idx": oracle_idx,
        "fqe_idx": fqe_idx,
        "oracle_return": oracle_return,
        "fqe_selected_true_return": selected_return,
        "selection_regret": float(oracle_return - selected_return),
        "top1_agreement": bool(fqe_idx == oracle_idx),
        # Backward-compatible direct-choice metrics from the previous study.
        "top3_hit": bool(fqe_idx in online_order[:min(3, n_candidates)]),
        "top5_hit": bool(fqe_idx in online_order[:min(5, n_candidates)]),
    }

    for requested_k in HYBRID_TOPK_VALUES:
        k = min(int(requested_k), n_candidates)
        shortlist = fqe_order[:k]
        oracle_recalled = bool(np.any(oracle_best_mask[shortlist]))

        shortlist_online = online_scores[shortlist]
        best_shortlist_pos = int(np.argmax(shortlist_online))
        hybrid_best_idx = int(shortlist[best_shortlist_pos])
        hybrid_best_return = float(online_scores[hybrid_best_idx])

        metrics[f"oracle_recall_at_{requested_k}"] = oracle_recalled
        metrics[f"hybrid_effective_k_at_{requested_k}"] = int(k)
        metrics[f"hybrid_best_idx_at_{requested_k}"] = hybrid_best_idx
        metrics[f"hybrid_best_true_return_at_{requested_k}"] = hybrid_best_return
        metrics[f"hybrid_regret_at_{requested_k}"] = float(
            oracle_return - hybrid_best_return
        )
        metrics[f"hybrid_online_fraction_at_{requested_k}"] = float(
            k / n_candidates
        )
        metrics[f"hybrid_online_reduction_at_{requested_k}"] = float(
            1.0 - (k / n_candidates)
        )

    return metrics



def compute_objective_mismatch_metrics(
    online_undiscounted,
    online_discounted,
    fqe_mean_q,
    iteration,
    gamma,
):
    """Compare raw FQE ranking against both online return definitions."""
    online_undiscounted = np.asarray(online_undiscounted, dtype=np.float64)
    online_discounted = np.asarray(online_discounted, dtype=np.float64)
    fqe_mean_q = np.asarray(fqe_mean_q, dtype=np.float64)

    if not (
        len(online_undiscounted)
        == len(online_discounted)
        == len(fqe_mean_q)
    ):
        raise RuntimeError(
            "Objective-mismatch length mismatch: "
            f"undiscounted={len(online_undiscounted)}, "
            f"discounted={len(online_discounted)}, FQE={len(fqe_mean_q)}."
        )
    if len(fqe_mean_q) == 0:
        raise RuntimeError("Objective-mismatch study received zero candidates.")

    def correlations(x, y):
        df = pd.DataFrame({"x": x, "y": y})
        return (
            float(df.corr(method="pearson").loc["x", "y"]),
            float(df.corr(method="spearman").loc["x", "y"]),
            float(df.corr(method="kendall").loc["x", "y"]),
        )

    undisc_corr = correlations(fqe_mean_q, online_undiscounted)
    disc_corr = correlations(fqe_mean_q, online_discounted)
    online_corr = correlations(online_undiscounted, online_discounted)

    n_candidates = len(fqe_mean_q)
    fqe_idx = int(np.argmax(fqe_mean_q))
    undisc_oracle_idx = int(np.argmax(online_undiscounted))
    disc_oracle_idx = int(np.argmax(online_discounted))
    undisc_order = np.argsort(online_undiscounted)[::-1]
    disc_order = np.argsort(online_discounted)[::-1]

    undisc_oracle_return = float(online_undiscounted[undisc_oracle_idx])
    disc_oracle_return = float(online_discounted[disc_oracle_idx])
    fqe_selected_undisc_return = float(online_undiscounted[fqe_idx])
    fqe_selected_disc_return = float(online_discounted[fqe_idx])

    metrics = {
        "iteration": int(iteration),
        "gamma": float(gamma),
        "n_candidates": int(n_candidates),
        "fqe_vs_undiscounted_pearson": undisc_corr[0],
        "fqe_vs_undiscounted_spearman": undisc_corr[1],
        "fqe_vs_undiscounted_kendall": undisc_corr[2],
        "fqe_vs_discounted_pearson": disc_corr[0],
        "fqe_vs_discounted_spearman": disc_corr[1],
        "fqe_vs_discounted_kendall": disc_corr[2],
        "discounted_minus_undiscounted_pearson": disc_corr[0] - undisc_corr[0],
        "discounted_minus_undiscounted_spearman": disc_corr[1] - undisc_corr[1],
        "discounted_minus_undiscounted_kendall": disc_corr[2] - undisc_corr[2],
        "online_objectives_pearson": online_corr[0],
        "online_objectives_spearman": online_corr[1],
        "online_objectives_kendall": online_corr[2],
        "fqe_idx": fqe_idx,
        "undiscounted_oracle_idx": undisc_oracle_idx,
        "discounted_oracle_idx": disc_oracle_idx,
        "online_oracle_top1_same": bool(undisc_oracle_idx == disc_oracle_idx),
        "fqe_top1_undiscounted": bool(fqe_idx == undisc_oracle_idx),
        "fqe_top1_discounted": bool(fqe_idx == disc_oracle_idx),
        "fqe_top3_undiscounted": bool(
            fqe_idx in undisc_order[:min(3, n_candidates)]
        ),
        "fqe_top3_discounted": bool(
            fqe_idx in disc_order[:min(3, n_candidates)]
        ),
        "fqe_top5_undiscounted": bool(
            fqe_idx in undisc_order[:min(5, n_candidates)]
        ),
        "fqe_top5_discounted": bool(
            fqe_idx in disc_order[:min(5, n_candidates)]
        ),
        "undiscounted_oracle_return": undisc_oracle_return,
        "discounted_oracle_return": disc_oracle_return,
        "fqe_selected_undiscounted_return": fqe_selected_undisc_return,
        "fqe_selected_discounted_return": fqe_selected_disc_return,
        "fqe_regret_undiscounted": float(
            undisc_oracle_return - fqe_selected_undisc_return
        ),
        "fqe_regret_discounted": float(
            disc_oracle_return - fqe_selected_disc_return
        ),
    }
    return metrics



def compute_fqe_convergence_metrics(
    online_scores,
    raw_fqe_scores,
    iteration,
    n_steps,
    target_update_interval,
    score_metadata=None,
):
    """Metrics for the full-replay raw-FQE convergence diagnostic.

    This intentionally ignores the behavioral support penalty. The purpose is
    to isolate whether the time-conditioned H-step Bellman objective has had
    enough optimization / target-network propagation to learn a useful value
    scale and ranking.
    """
    online_scores = np.asarray(online_scores, dtype=np.float64)
    raw_fqe_scores = np.asarray(raw_fqe_scores, dtype=np.float64)

    if online_scores.shape != raw_fqe_scores.shape:
        raise RuntimeError(
            "FQE convergence length mismatch: "
            f"online={online_scores.shape}, FQE={raw_fqe_scores.shape}."
        )
    if len(online_scores) == 0:
        raise RuntimeError("FQE convergence study received zero candidates.")
    if not np.all(np.isfinite(raw_fqe_scores)):
        raise RuntimeError("FQE convergence study received non-finite Q values.")

    rank_df = pd.DataFrame({"fqe": raw_fqe_scores, "online": online_scores})
    pearson = float(rank_df.corr(method="pearson").loc["fqe", "online"])
    spearman = float(rank_df.corr(method="spearman").loc["fqe", "online"])
    kendall = float(rank_df.corr(method="kendall").loc["fqe", "online"])

    n_candidates = len(online_scores)
    oracle_idx = int(np.argmax(online_scores))
    fqe_idx = int(np.argmax(raw_fqe_scores))
    online_order = np.argsort(online_scores)[::-1]
    oracle_return = float(online_scores[oracle_idx])
    selected_return = float(online_scores[fqe_idx])

    n_steps = int(n_steps)
    target_update_interval = int(target_update_interval)
    n_target_updates = int(((n_steps - 1) // target_update_interval) + 1)

    metadata = score_metadata if score_metadata is not None else object()
    finite_horizon_steps = getattr(metadata, "finite_horizon_steps", None)

    return {
        "iteration": int(iteration),
        "config": f"{n_steps}_steps_target{target_update_interval}",
        "fqe_n_steps": n_steps,
        "fqe_target_update_interval": target_update_interval,
        "fqe_target_updates": n_target_updates,
        "fqe_objective": str(getattr(metadata, "fqe_objective", "unknown")),
        "fqe_gamma": float(getattr(metadata, "fqe_gamma", np.nan)),
        "finite_horizon_steps": (
            -1 if finite_horizon_steps is None else int(finite_horizon_steps)
        ),
        "time_conditioned": bool(getattr(metadata, "time_conditioned", False)),
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "oracle_idx": oracle_idx,
        "fqe_idx": fqe_idx,
        "top1_agreement": bool(fqe_idx == oracle_idx),
        "top3_hit": bool(fqe_idx in online_order[:min(3, n_candidates)]),
        "top5_hit": bool(fqe_idx in online_order[:min(5, n_candidates)]),
        "oracle_return": oracle_return,
        "fqe_selected_true_return": selected_return,
        "selection_regret": float(oracle_return - selected_return),
        # Q-scale / compression diagnostics. Absolute calibration is not a
        # ranking requirement, but these values show whether the long-horizon
        # Bellman target is still severely under-propagated.
        "mean_fqe_q": float(np.mean(raw_fqe_scores)),
        "std_fqe_q": float(np.std(raw_fqe_scores, ddof=0)),
        "min_fqe_q": float(np.min(raw_fqe_scores)),
        "max_fqe_q": float(np.max(raw_fqe_scores)),
        "fqe_q_range": float(np.ptp(raw_fqe_scores)),
        "mean_online_return": float(np.mean(online_scores)),
        "std_online_return": float(np.std(online_scores, ddof=0)),
        "online_return_range": float(np.ptp(online_scores)),
        "mean_final_fqe_loss": float(
            getattr(metadata, "mean_final_loss", np.nan)
        ),
    }


def compute_knn_support_filter_metrics(
    online_scores,
    raw_fqe_scores,
    support_diagnostics,
    iteration,
    score_metadata=None,
):
    """Evaluate the state-conditional support FILTER without changing selection.

    The absolute threshold is defined only by frozen replay behavior. If fewer
    than KNN_SUPPORT_MIN_KEEP candidates pass, the pool is augmented with the
    most-supported candidates, ordered by unsupported fraction then mean local
    action distance. Online returns are NEVER used to define the filter.
    Within the resulting pool, raw 50k/100 FQE determines the ranking.
    """
    online_scores = np.asarray(online_scores, dtype=np.float64)
    raw_fqe_scores = np.asarray(raw_fqe_scores, dtype=np.float64)
    unsupported = np.asarray(
        support_diagnostics["candidate_unsupported_fraction"], dtype=np.float64
    )
    mean_distance = np.asarray(
        support_diagnostics["candidate_mean_sq_l2"], dtype=np.float64
    )

    n_candidates = len(online_scores)
    if not (
        raw_fqe_scores.shape == online_scores.shape
        == unsupported.shape == mean_distance.shape
    ):
        raise RuntimeError(
            "kNN support filter length mismatch: "
            f"online={online_scores.shape}, FQE={raw_fqe_scores.shape}, "
            f"unsupported={unsupported.shape}, distance={mean_distance.shape}."
        )
    if n_candidates == 0:
        raise RuntimeError("kNN support filter received zero candidates.")
    if not (
        np.all(np.isfinite(raw_fqe_scores))
        and np.all(np.isfinite(unsupported))
        and np.all(np.isfinite(mean_distance))
    ):
        raise RuntimeError("kNN support filter received non-finite diagnostics.")

    oracle_idx = int(np.argmax(online_scores))
    oracle_return = float(online_scores[oracle_idx])
    oracle_best_mask = online_scores == oracle_return
    online_order = np.argsort(online_scores)[::-1]
    raw_fqe_order = np.argsort(raw_fqe_scores)[::-1]
    raw_fqe_idx = int(raw_fqe_order[0])

    threshold_pass = (
        unsupported <= KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION
    )
    effective_keep = threshold_pass.copy()

    # Pure support ordering; FQE and online returns do not participate in the
    # minimum-pool fallback.
    support_order = np.lexsort((mean_distance, unsupported))
    min_keep = min(int(KNN_SUPPORT_MIN_KEEP), n_candidates)
    if int(np.sum(effective_keep)) < min_keep:
        for candidate_idx in support_order:
            effective_keep[int(candidate_idx)] = True
            if int(np.sum(effective_keep)) >= min_keep:
                break

    threshold_keep_count = int(np.sum(threshold_pass))
    effective_keep_count = int(np.sum(effective_keep))
    fallback_fill_count = int(effective_keep_count - threshold_keep_count)
    kept_indices = np.flatnonzero(effective_keep)
    if len(kept_indices) == 0:
        raise RuntimeError("kNN support filter produced an empty candidate pool.")

    kept_order = kept_indices[
        np.argsort(raw_fqe_scores[kept_indices])[::-1]
    ]
    filtered_idx = int(kept_order[0])
    most_supported_idx = int(support_order[0])

    def _corr(x, y, method):
        frame = pd.DataFrame({"x": x, "y": y})
        return float(frame.corr(method=method).loc["x", "y"])

    metadata = score_metadata if score_metadata is not None else object()
    finite_horizon_steps = getattr(metadata, "finite_horizon_steps", None)

    metrics = {
        "iteration": int(iteration),
        "fqe_config": (
            f"{int(getattr(metadata, 'fqe_n_steps', KNN_SUPPORT_FQE_N_STEPS))}"
            f"_steps_target"
            f"{int(getattr(metadata, 'fqe_target_update_interval', KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL))}"
        ),
        "fqe_n_steps": int(
            getattr(metadata, "fqe_n_steps", KNN_SUPPORT_FQE_N_STEPS)
        ),
        "fqe_target_update_interval": int(
            getattr(
                metadata,
                "fqe_target_update_interval",
                KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL,
            )
        ),
        "fqe_objective": str(getattr(metadata, "fqe_objective", "unknown")),
        "fqe_gamma": float(getattr(metadata, "fqe_gamma", np.nan)),
        "finite_horizon_steps": (
            -1 if finite_horizon_steps is None else int(finite_horizon_steps)
        ),
        "time_conditioned": bool(getattr(metadata, "time_conditioned", False)),
        "reference_transitions": int(support_diagnostics["reference_count"]),
        "query_states": int(support_diagnostics["query_count"]),
        "knn_k": int(support_diagnostics["k"]),
        "behavior_percentile": float(
            support_diagnostics["behavior_percentile"]
        ),
        "behavior_threshold_sq_l2": float(
            support_diagnostics["behavior_threshold_sq_l2"]
        ),
        "behavior_mean_sq_l2": float(
            support_diagnostics["behavior_mean_sq_l2"]
        ),
        "behavior_p95_sq_l2": float(
            support_diagnostics["behavior_p95_sq_l2"]
        ),
        "mean_neighbor_state_distance": float(
            support_diagnostics["mean_neighbor_state_distance"]
        ),
        "p95_neighbor_state_distance": float(
            support_diagnostics["p95_neighbor_state_distance"]
        ),
        "max_unsupported_fraction": float(
            KNN_SUPPORT_MAX_UNSUPPORTED_FRACTION
        ),
        "min_keep": int(min_keep),
        "threshold_keep_count": threshold_keep_count,
        "effective_keep_count": effective_keep_count,
        "fallback_fill_count": fallback_fill_count,
        "oracle_idx": oracle_idx,
        "oracle_return": oracle_return,
        "oracle_passes_absolute_filter": bool(threshold_pass[oracle_idx]),
        "oracle_survives_effective_filter": bool(effective_keep[oracle_idx]),
        "raw_fqe_idx": raw_fqe_idx,
        "raw_fqe_selected_true_return": float(online_scores[raw_fqe_idx]),
        "raw_fqe_selection_regret": float(
            oracle_return - online_scores[raw_fqe_idx]
        ),
        "raw_fqe_top1": bool(raw_fqe_idx == oracle_idx),
        "raw_fqe_top3": bool(raw_fqe_idx in online_order[:min(3, n_candidates)]),
        "raw_fqe_top5": bool(raw_fqe_idx in online_order[:min(5, n_candidates)]),
        "filtered_fqe_idx": filtered_idx,
        "filtered_fqe_selected_true_return": float(online_scores[filtered_idx]),
        "filtered_fqe_selection_regret": float(
            oracle_return - online_scores[filtered_idx]
        ),
        "filtered_fqe_top1": bool(filtered_idx == oracle_idx),
        "filtered_fqe_top3": bool(
            filtered_idx in online_order[:min(3, n_candidates)]
        ),
        "filtered_fqe_top5": bool(
            filtered_idx in online_order[:min(5, n_candidates)]
        ),
        "most_supported_idx": most_supported_idx,
        "most_supported_true_return": float(online_scores[most_supported_idx]),
        "most_supported_regret": float(
            oracle_return - online_scores[most_supported_idx]
        ),
        "raw_fqe_pearson": _corr(raw_fqe_scores, online_scores, "pearson"),
        "raw_fqe_spearman": _corr(raw_fqe_scores, online_scores, "spearman"),
        "raw_fqe_kendall": _corr(raw_fqe_scores, online_scores, "kendall"),
        # Higher support score means better support, hence the minus sign.
        "support_score_spearman": _corr(-unsupported, online_scores, "spearman"),
        "support_distance_spearman": _corr(-mean_distance, online_scores, "spearman"),
        "fqe_vs_support_spearman": _corr(
            raw_fqe_scores, -unsupported, "spearman"
        ),
        "mean_candidate_unsupported_fraction": float(np.mean(unsupported)),
        "min_candidate_unsupported_fraction": float(np.min(unsupported)),
        "max_candidate_unsupported_fraction": float(np.max(unsupported)),
        "mean_candidate_local_sq_l2": float(np.mean(mean_distance)),
    }

    for requested_k in HYBRID_TOPK_VALUES:
        raw_k = min(int(requested_k), n_candidates)
        raw_shortlist = raw_fqe_order[:raw_k]
        raw_best_pos = int(np.argmax(online_scores[raw_shortlist]))
        raw_hybrid_idx = int(raw_shortlist[raw_best_pos])
        raw_hybrid_return = float(online_scores[raw_hybrid_idx])

        filtered_k = min(int(requested_k), len(kept_order))
        filtered_shortlist = kept_order[:filtered_k]
        filtered_best_pos = int(np.argmax(online_scores[filtered_shortlist]))
        filtered_hybrid_idx = int(filtered_shortlist[filtered_best_pos])
        filtered_hybrid_return = float(online_scores[filtered_hybrid_idx])

        metrics[f"raw_oracle_recall_at_{requested_k}"] = bool(
            np.any(oracle_best_mask[raw_shortlist])
        )
        metrics[f"raw_hybrid_regret_at_{requested_k}"] = float(
            oracle_return - raw_hybrid_return
        )
        metrics[f"filtered_effective_k_at_{requested_k}"] = int(filtered_k)
        metrics[f"filtered_oracle_recall_at_{requested_k}"] = bool(
            np.any(oracle_best_mask[filtered_shortlist])
        )
        metrics[f"filtered_hybrid_regret_at_{requested_k}"] = float(
            oracle_return - filtered_hybrid_return
        )
        metrics[f"filtered_online_reduction_at_{requested_k}"] = float(
            1.0 - (filtered_k / n_candidates)
        )

    # Per-candidate rank annotations for CSV analysis / post-hoc threshold sweeps.
    support_rank = np.empty(n_candidates, dtype=np.int64)
    support_rank[support_order] = np.arange(1, n_candidates + 1, dtype=np.int64)
    raw_fqe_rank = np.empty(n_candidates, dtype=np.int64)
    raw_fqe_rank[raw_fqe_order] = np.arange(1, n_candidates + 1, dtype=np.int64)
    filtered_rank = np.full(n_candidates, -1, dtype=np.int64)
    filtered_rank[kept_order] = np.arange(1, len(kept_order) + 1, dtype=np.int64)

    candidate_details = {
        "threshold_pass": threshold_pass.astype(bool),
        "effective_keep": effective_keep.astype(bool),
        "support_rank": support_rank,
        "raw_fqe_rank": raw_fqe_rank,
        "filtered_fqe_rank": filtered_rank,
    }
    return metrics, candidate_details


def replay_buffer_signature(replay_buffer):
    """Small invariant used to detect accidental candidate-evaluation data leakage."""
    return (
        getattr(replay_buffer, "pos", None),
        getattr(replay_buffer, "full", None),
        tuple(replay_buffer.observations.shape),
        tuple(replay_buffer.actions.shape),
        tuple(replay_buffer.rewards.shape),
        tuple(replay_buffer.dones.shape),
        tuple(replay_buffer.timeouts.shape),
    )


def d3rl_evaluation_preserving_rng(model, exp_name, dataset):
    """Run analysis-only FQE without perturbing the online oracle's RNG trajectory."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        return d3rl_evaluation(model, exp_name, dataset=dataset)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


# Method for evaluating the FQE using d3rlpy
def d3rl_evaluation(model, exp_name, dataset=None):
    # When dataset is supplied by the rank-correlation study, all candidates in the current
    # iteration use one frozen offline dataset. Without it, retain the original behavior.
    if dataset is None:
        dataset = build_fqe_dataset(model)

    try:
        ppo_wrapper = PPOQWrapper(model)
        ppo_wrapper.build_with_dataset(dataset)

    except Exception as e:
        print(f"Error creating or building the PPOQWrapper: {e}")
        import traceback
        traceback.print_exc()
        exit()

    try:
        fqe = d3rlpy.ope.FQE(
            algo=ppo_wrapper,
            config=d3rlpy.ope.FQEConfig(
                learning_rate=3e-4,
                # Preserve the original d3rlpy fallback semantics exactly.
                # The convergence-study target-update knob is native-FQE only.
                target_update_interval=100,
                gamma=ppo_wrapper.ppo.gamma,
                batch_size=FQE_BATCH_SIZE,
            ),
            device=str(device),
        )

        print("--------------------------------------------------------------------------------")
        print("Fitting d3rlpy FQE...")
        print(
            "FQE runtime config: "
            f"steps={FQE_N_STEPS}, "
            f"steps_per_epoch={FQE_N_STEPS_PER_EPOCH}, "
            f"batch_size={FQE_BATCH_SIZE}, "
            f"progress={FQE_SHOW_PROGRESS}, "
            f"file_logging={FQE_USE_FILE_LOGGER}"
        )

        # Default d3rlpy FileAdapter writes metrics/checkpoints for every fit.
        # The rank study needs only the final scalar score, so use a no-op
        # adapter by default. This changes I/O only, not FQE optimization.
        if FQE_USE_FILE_LOGGER:
            logger_adapter = d3rlpy.logging.FileAdapterFactory()
        else:
            logger_adapter = d3rlpy.logging.NoopAdapterFactory()

        fit_start = time.time()

        # Preserve the same number of gradient updates and the same epoch length.
        # The expensive InitialStateValue evaluator is no longer run after every
        # epoch because only its final value was ever consumed.
        fqe.fit(
            dataset,
            n_steps=FQE_N_STEPS,
            n_steps_per_epoch=FQE_N_STEPS_PER_EPOCH,
            evaluators=None,
            show_progress=FQE_SHOW_PROGRESS,
            logger_adapter=logger_adapter,
            # With NoopAdapter this performs no disk write. Keeping a valid
            # positive interval maintains compatibility across d3rlpy versions.
            save_interval=max(1, FQE_N_STEPS // FQE_N_STEPS_PER_EPOCH),
            experiment_name=exp_name,
        )

        fit_seconds = time.time() - fit_start
        print(f"FQE fitting time: {fit_seconds:.3f} s")
        print()
        print("FQE Fitting completed.")

        # Compute the same reported metric once after the final update.
        init_evaluator = d3rlpy.metrics.InitialStateValueEstimationEvaluator()
        initial_state_value = init_evaluator(fqe, dataset)

        print(f"Estimated Initial State Value: {initial_state_value}")
        return initial_state_value

    except Exception as e:
        print(f"Error during FQE configuration or fitting: {e}")
        import traceback
        traceback.print_exc()

# Function to average multiple checkpoints
def average_checkpoints(checkpoint_paths):
    """Averages the model weights from a list of checkpoint paths."""
    policies = []
    for path in checkpoint_paths:
        policy_vec = []
        ckp = torch.load(path, map_location='cpu')
        ckp_layers = ckp.keys()

        for layer in ckp_layers:
            if 'value_net' not in layer:
                policy_vec.append(ckp[layer].detach().numpy().reshape(-1))

        policy_vec = np.concatenate(policy_vec)
        policies.append(policy_vec)

    policies = np.array(policies)
    avg_policy_vec = np.mean(policies, axis=0)
    
    return avg_policy_vec

# Guided Evolutionary Strategies
def search_guided_es_policies(algo, directory, start, end, env, saved_agents, agent_num=10, sigma=0.02, alpha=0.5, num_candidates=20):
    print("---------------------------------")
    print("Searching policies using Guided Evolutionary Strategies")

    # Load current policy parameters as flat vector
    policy_state = algo.policy.state_dict()
    theta_anchor = []
    for layer in policy_state:
        if 'value_net' not in layer:
            theta_anchor.append(policy_state[layer].detach().cpu().numpy().reshape(-1))
    theta_anchor = np.concatenate(theta_anchor)
    theta_dim = theta_anchor.shape[0]

    # Get PPO gradient (already computed during model.learn)
    algo.policy.zero_grad()
    dummy_env = DummyVecEnv([lambda: gym.make(env_name)])
    obs = dummy_env.reset()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
    obs_tensor = obs_tensor.unsqueeze(0)  # [1, obs_dim]
    distribution = algo.policy.get_distribution(obs_tensor)
    action = distribution.sample()
    log_prob = distribution.log_prob(action).sum(dim=-1)
    log_prob.mean().backward()  # dummy backward pass to populate gradients

    ppo_grad = []
    for name, param in algo.policy.named_parameters():
        if 'value_net' in name:
            continue  # Skip critic parameters
        if param.grad is not None:
            ppo_grad.append(param.grad.view(-1).cpu().numpy())
    ppo_grad = np.concatenate(ppo_grad)

    # Generate ES candidates and accumulate gradients
    g_es_total = np.zeros_like(theta_anchor)

    for _ in range(num_candidates):
        epsilon = np.random.randn(theta_dim)
        theta_plus = theta_anchor + sigma * epsilon
        theta_minus = theta_anchor - sigma * epsilon

        # Evaluate both perturbations
        candidates = [theta_plus, theta_minus]
        rewards = []
        for theta in candidates:
            policy = OrderedDict()
            pivot = 0
            for layer in policy_state:
                if 'value_net' in layer:
                    policy[layer] = policy_state[layer]
                else:
                    sp = policy_state[layer].reshape(-1).shape[0]
                    policy[layer] = torch.as_tensor(
                        theta[pivot:pivot + sp].reshape(policy_state[layer].shape),
                        dtype=policy_state[layer].dtype,
                        device=policy_state[layer].device,
                    )
                    pivot += sp
            algo.policy.load_state_dict(policy)
            algo.policy.to(device)
            R = evaluate_policy(algo, dummy_env, n_eval_episodes=3, deterministic=True)[0]
            rewards.append(R)

        R_plus, R_minus = rewards
        g_es = ((R_plus - R_minus) / (2 * sigma)) * epsilon
        g_es_total += g_es

    g_es_mean = g_es_total / num_candidates
    g_combined = alpha * ppo_grad + (1 - alpha) * g_es_mean
    theta_new = theta_anchor + algo.learning_rate * g_combined

    # Convert updated flat vector back into policy state dict
    new_policy = OrderedDict()
    pivot = 0
    for layer in policy_state:
        if 'value_net' in layer:
            new_policy[layer] = policy_state[layer]
        else:
            sp = policy_state[layer].reshape(-1).shape[0]
            new_policy[layer] = torch.as_tensor(
                theta_new[pivot:pivot + sp].reshape(policy_state[layer].shape),
                dtype=policy_state[layer].dtype,
                device=policy_state[layer].device,
            )
            pivot += sp

    agent_list = [new_policy]  # we return one updated agent
    close_env_safely(dummy_env)
    return agent_list, 0.0  # dummy distance value for compatibility

# Value Function Search
def search_vfs_policies(algo, directory, start, end, env, saved_agents, agent_num=10,
                        alpha=0.01, vfs_steps=3):
    print("---------------------------------")
    print("Searching policies using Value Function Search")

    # Clone original policy weights to restore later if needed
    original_state = copy.deepcopy(algo.policy.state_dict())

    # Get one observation to condition value function
    dummy_env = DummyVecEnv([lambda: gym.make(env_name)])
    obs = dummy_env.reset()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)

    # Perform k VFS gradient ascent steps
    for step in range(vfs_steps):
        algo.policy.zero_grad()
        value = algo.policy.predict_values(obs_tensor).mean()
        value.backward()

        with torch.no_grad():
            for name, param in algo.policy.named_parameters():
                if 'value_net' in name or param.grad is None:
                    continue
                param += alpha * param.grad

    # Store the updated policy
    updated_state = copy.deepcopy(algo.policy.state_dict())
    agent_list = [updated_state]

    # Restore original policy weights to avoid affecting main model
    algo.policy.load_state_dict(original_state)
    close_env_safely(dummy_env)

    return agent_list, 0.0

class DiscountedReturnTracker:
    """Collect discounted returns and optional read-only online state traces.

    Stable-Baselines3 calls the evaluation callback once per active VecEnv slot
    after each environment step. Discounted returns are accumulated exactly as
    before. When ``capture_observations`` is enabled, the callback also copies
    the PRE-STEP observation from those SAME online evaluation transitions.
    The callback never writes to evaluator locals and never touches PPO replay.
    """

    def __init__(self, gamma, capture_observations=False):
        self.gamma = float(gamma)
        if not np.isfinite(self.gamma) or self.gamma < 0.0:
            raise ValueError("Discount gamma must be finite and >= 0.")
        self.capture_observations = bool(capture_observations)
        self.running_returns = {}
        self.discount_powers = {}
        self.episode_returns = []
        self.episode_lengths = []
        self.running_lengths = {}
        self.running_observations = {}
        self.trajectory_episodes = []

    def _capture_current_observation(self, local_vars, env_idx):
        if not self.capture_observations:
            return

        if "observations" not in local_vars:
            raise RuntimeError(
                "evaluate_policy callback did not expose the pre-step "
                "'observations' variable required by the state-occupancy "
                "diagnostic."
            )

        observations = local_vars["observations"]
        if isinstance(observations, dict):
            raise NotImplementedError(
                "State-occupancy online tracing currently expects flat vector "
                "observations, not Dict observations."
            )

        observations = np.asarray(observations)
        if observations.ndim == 1:
            if env_idx != 0:
                raise RuntimeError(
                    "State-occupancy callback received an unbatched observation "
                    f"for env_idx={env_idx}."
                )
            current_observation = observations
        else:
            if not (0 <= env_idx < observations.shape[0]):
                raise RuntimeError(
                    "State-occupancy callback env index is outside the "
                    f"observation batch: env_idx={env_idx}, "
                    f"shape={observations.shape}."
                )
            current_observation = observations[env_idx]

        current_observation = np.asarray(
            current_observation, dtype=np.float32
        ).reshape(-1)
        if not np.all(np.isfinite(current_observation)):
            raise RuntimeError(
                "State-occupancy callback observed non-finite online state."
            )

        self.running_observations.setdefault(env_idx, []).append(
            current_observation.copy()
        )

    def __call__(self, local_vars, global_vars):
        # Standard SB3 evaluate_policy exposes the current VecEnv slot as ``i``.
        # Keep a defensive scalar-env fallback for compatible custom evaluators.
        env_idx = int(local_vars.get("i", 0))

        # Copy the PRE-STEP state before episode-boundary bookkeeping. This is
        # the state on which the deterministic candidate action was chosen.
        self._capture_current_observation(local_vars, env_idx)

        if "reward" in local_vars:
            reward = float(np.asarray(local_vars["reward"]).reshape(-1)[0])
        elif "rewards" in local_vars:
            rewards = np.asarray(local_vars["rewards"]).reshape(-1)
            reward = float(rewards[env_idx])
        else:
            raise RuntimeError(
                "evaluate_policy callback did not expose reward(s); cannot "
                "compute discounted online return on the same trajectory."
            )

        if "done" in local_vars:
            done = bool(np.asarray(local_vars["done"]).reshape(-1)[0])
        elif "dones" in local_vars:
            dones = np.asarray(local_vars["dones"]).reshape(-1)
            done = bool(dones[env_idx])
        else:
            raise RuntimeError(
                "evaluate_policy callback did not expose done(s); cannot "
                "detect discounted-return episode boundaries."
            )

        running_return = self.running_returns.get(env_idx, 0.0)
        discount_power = self.discount_powers.get(env_idx, 1.0)
        running_length = self.running_lengths.get(env_idx, 0)

        running_return += discount_power * reward
        discount_power *= self.gamma
        running_length += 1

        self.running_returns[env_idx] = running_return
        self.discount_powers[env_idx] = discount_power
        self.running_lengths[env_idx] = running_length

        if done:
            # Mirror evaluate_policy's episode-counting semantics. With a
            # Monitor/VecMonitor wrapper, ``done`` can occur without a true
            # counted episode (e.g. Atari life loss); SB3 counts the episode
            # only when Monitor supplies info["episode"].
            is_monitor_wrapped = bool(
                local_vars.get("is_monitor_wrapped", False)
            )
            info = local_vars.get("info", {})
            counted_episode = (
                not is_monitor_wrapped
                or (isinstance(info, dict) and "episode" in info)
            )
            if counted_episode:
                self.episode_returns.append(float(running_return))
                self.episode_lengths.append(int(running_length))

                if self.capture_observations:
                    states = self.running_observations.get(env_idx, [])
                    if len(states) != int(running_length):
                        raise RuntimeError(
                            "State-occupancy callback state-count mismatch: "
                            f"states={len(states)}, steps={running_length}."
                        )
                    self.trajectory_episodes.append(
                        np.asarray(states, dtype=np.float32)
                    )

            # VecEnv resets the underlying environment on done regardless of
            # whether Monitor treats that boundary as a counted episode.
            self.running_returns[env_idx] = 0.0
            self.discount_powers[env_idx] = 1.0
            self.running_lengths[env_idx] = 0
            if self.capture_observations:
                self.running_observations[env_idx] = []



def evaluate_policy_with_discounted_return(
    model,
    env,
    n_eval_episodes,
    gamma,
    deterministic=True,
    capture_trajectory=False,
    **evaluate_kwargs,
):
    """Run the original SB3 online evaluation with read-only diagnostics.

    The undiscounted result is returned verbatim from ``evaluate_policy`` and is
    still the quantity used by the original selector. The discounted return is
    shadowed by the callback exactly as before. When ``capture_trajectory`` is
    true, copies of the pre-step observations from those SAME evaluation
    episodes are returned as an additional diagnostic value.

    Captured states are never added to PPO replay and never used to train FQE.
    """
    if "callback" in evaluate_kwargs:
        raise ValueError(
            "evaluate_policy_with_discounted_return owns the callback so the "
            "discounted/state-occupancy diagnostics cannot be mixed with "
            "another callback."
        )

    tracker = DiscountedReturnTracker(
        gamma=gamma,
        capture_observations=capture_trajectory,
    )
    result = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        callback=tracker,
        **evaluate_kwargs,
    )

    if len(tracker.episode_returns) != int(n_eval_episodes):
        raise RuntimeError(
            "Discounted-return callback observed "
            f"{len(tracker.episode_returns)} completed episodes, expected "
            f"{int(n_eval_episodes)}. The local Stable-Baselines3 evaluator "
            "callback semantics may differ from the expected API."
        )

    discounted_mean = float(np.mean(tracker.episode_returns))
    episode_returns = np.asarray(
        tracker.episode_returns, dtype=np.float64
    )

    if not capture_trajectory:
        # Preserve the exact historical 3-value return contract.
        return result, discounted_mean, episode_returns

    if len(tracker.trajectory_episodes) != int(n_eval_episodes):
        raise RuntimeError(
            "State-occupancy callback observed "
            f"{len(tracker.trajectory_episodes)} completed trajectory traces, "
            f"expected {int(n_eval_episodes)}."
        )

    trajectory_episodes = [
        np.asarray(states, dtype=np.float32).copy()
        for states in tracker.trajectory_episodes
    ]
    return result, discounted_mean, episode_returns, trajectory_episodes


# Rollout policy to get average reward
# gamma=None preserves the original return type and behavior.
def rollout_policy(policy, env, n_eval=3, deterministic=True, gamma=None):
    episode_rewards = []
    discounted_episode_rewards = []

    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        discounted_reward = 0.0
        discount_power = 1.0

        while not done:
            # SB3 uses: policy.predict(obs, deterministic)
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            if gamma is not None:
                discounted_reward += discount_power * reward
                discount_power *= float(gamma)

        episode_rewards.append(total_reward)
        if gamma is not None:
            discounted_episode_rewards.append(discounted_reward)

    # Preserve the original return value when no discounted diagnostic is asked for.
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    if gamma is None:
        return mean_reward

    mean_discounted_reward = (
        sum(discounted_episode_rewards) / len(discounted_episode_rewards)
    )
    return mean_reward, mean_discounted_reward

# Evaluation function for a single candidate agent
def evaluate_candidate(args):
    idx, agent_state_dict, env_name, seed, n_eval, gamma = args

    dummy_env = gym.make(env_name)
    dummy_env.reset(seed=seed)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    # Build policy network only (not PPO)
    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 0.0
    )
    policy.load_state_dict(agent_state_dict)

    try:
        rollout_result = rollout_policy(
            policy,
            dummy_env,
            n_eval=n_eval,
            deterministic=True,
            gamma=gamma,
        )
    finally:
        close_env_safely(dummy_env)

    if gamma is None:
        avg_return = rollout_result
        avg_discounted_return = None
    else:
        avg_return, avg_discounted_return = rollout_result

    # Evaluate policy
    print(f"avg return on {n_eval} trajectories of agent{idx}: {avg_return}")
    if avg_discounted_return is not None:
        print(
            f"avg gamma-discounted return on {n_eval} trajectories of "
            f"agent{idx}: {avg_discounted_return}"
        )
    return idx, avg_return, avg_discounted_return

# Parallel Evaluation of multiple agents
def parallel_evaluate(
    agents,
    env_name,
    seed,
    n_eval_episodes=3,
    gamma=None,
    return_discounted=False,
):
    print("Evaluating", len(agents), "agents in parallel...")

    # Prepare job arguments
    cpu_agents = agents_to_cpu(agents)
    job_args = [
        (
            j,
            cpu_agents[j],
            env_name,
            seed,
            n_eval_episodes,
            gamma if return_discounted else None,
        )
        for j in range(len(cpu_agents))
    ]

    # Number of worker processes
    n_workers = min(len(agents), cpu_count())

    with Pool(processes=n_workers) as pool:
        results = pool.map(evaluate_candidate, job_args)

    # Sort by index
    results = sorted(results, key=lambda x: x[0])

    returns = [r[1] for r in results]
    if not return_discounted:
        # Backward-compatible path.
        return returns

    discounted_returns = [r[2] for r in results]
    return returns, discounted_returns

# ------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    args, rest_args = parser.parse_known_args()

    env_name = "Ant-v5" # For standard ant locomotion task (single goal task)
    # env_name = "HalfCheetah-v5" # For standard half-cheetah locomotion task (single goal task)
    # env_name = "Hopper-v5" # For standard hopper locomotion task (single goal task)
    # env_name = "Walker2d-v5" # For standard walker locomotion task (single goal task)
    # env_name = "Humanoid-v5" # For standard ant locomotion task (single goal task)
    # env_name = "Swimmer-v5" # For standard swimmer locomotion task (single goal task)

    # env_name = "CartPole-v1" # For cartpole (single goal task)
    # env_name = "MountainCar-v0" # For mountain car (single goal task)
    # env_name = "Pendulum-v1" # For pendulum (single goal task)

    # env_name = "FetchReach-v4" # For FetchReach (single goal task) sparse rewards
    # env_name = "FetchReachDense-v4" # For FetchReach (single goal task) dense rewards
    # env_name = "FetchPush-v4" # For FetchPush (single goal task) sparse rewards
    # env_name = "FetchPushDense-v4" # For FetchPush (single goal task) dense rewards

    # env_name = "BreakoutNoFrameskip-v4" # For Breakout Atari (single goal task)

    # env_name = "AntDir-v0" # Part of the Meta-World or Meta-RL (meta-reinforcement learning) benchmarks (used for multi-task learning)

    if env_name == "AntDir-v0":
        args = args_ant_dir.get_args(rest_args)
    elif env_name == "Ant-v5":
        args = args_ant.get_args(rest_args)
    elif env_name == "Hopper-v5":
        args = args_hopper.get_args(rest_args)
    elif env_name == "HalfCheetah-v5":
        args = args_half_cheetah.get_args(rest_args)
    elif env_name == "Walker2d-v5":
        args = args_walker2d.get_args(rest_args)
    elif env_name == "Humanoid-v5":
        args = args_humanoid.get_args(rest_args)
    elif env_name == "Swimmer-v5":
        args = args_swimmer.get_args(rest_args)
    elif env_name == "CartPole-v1":
        args = args_cartpole.get_args(rest_args)
    elif env_name == "MountainCar-v0":
        args = args_mountain_car.get_args(rest_args)
    elif env_name == "Pendulum-v1":
        args = args_pendulum.get_args(rest_args)
    elif env_name == "FetchReach-v4":
        args = args_fetch_reach.get_args(rest_args)
    elif env_name == "FetchReachDense-v4":
        args = args_fetch_reach_dense.get_args(rest_args)
    elif env_name == "FetchPush-v4":
        args = args_fetch_push.get_args(rest_args)
    elif env_name == "FetchPushDense-v4":
        args = args_fetch_push_dense.get_args(rest_args)
    elif env_name == "BreakoutNoFrameskip-v4":
        args = args_breakout_no_frameskip.get_args(rest_args)

    if TIME_CONDITIONED_FINITE_HORIZON_FQE:
        fqe_finite_horizon_steps = resolve_fqe_finite_horizon(env_name)
        print(
            "Time-conditioned finite-horizon FQE enabled: "
            f"H={fqe_finite_horizon_steps}, OPE gamma=1.0, "
            "critic input includes normalized t/H."
        )
    else:
        fqe_finite_horizon_steps = None
        print(
            "Time-conditioned finite-horizon FQE disabled: "
            "native FQE keeps the original discounted objective."
        )

    # ------------------------------------------------------------------------------------------------------------
    # Force the training/inference device to match the CUDA auto-detection above. This keeps one
    # consistent device across PPO, generated policy state dicts, and the GPU empty-space search.
    args.device = str(device)
    print_device_info()

    # Set the seed for reproducibility
    if hasattr(args, 'seed'):
        print("Setting seed - ", args.seed)
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        np.random.seed(args.seed)

    # -------------------------------------------------------------------------------------------------------------
    
    def make_envs(env_name, seed):
        def _init(seed_offset):
            def _thunk():
                env = gym.make(env_name)
                env.reset(seed=seed + seed_offset)
                return env
            return _thunk
        return _init
    
    if hasattr(args, 'n_envs') and args.n_envs > 1:
        print("Creating multiple envs - ", args.n_envs)
        if env_name in ["BreakoutNoFrameskip-v4"]:
            env = make_atari_env(
                env_name,            # e.g. "BreakoutNoFrameskip-v4"
                n_envs=args.n_envs,
                seed=args.seed,
                wrapper_kwargs=dict(terminal_on_life_loss=False),
            )
        else:
            # Create a list of environment functions
            env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=i) for i in range(args.n_envs)]
            env = SubprocVecEnv(env_fns)
    else:
        if env_name in ["BreakoutNoFrameskip-v4"]:
            env = make_atari_env(
                env_name,
                n_envs=1,
                seed=args.seed,
                wrapper_kwargs=dict(terminal_on_life_loss=False),
            )
        else:
            env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5
            env.reset(seed=args.seed)
    
    if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
        env = FlattenObservation(env)
    elif env_name in ["BreakoutNoFrameskip-v4"]:
        env = VecFrameStack(env, n_stack=4)

    # env = make_env(env_name, episodes_per_task=1, seed=0, n_tasks=1) # For AntDir-v0

    print("---------------------------------")
    print("Environment created")
    print(env.action_space, env.observation_space)
    
    # ------------------------------------------------------------------------------------------------------------
    # goal = np.random.uniform(0, 3.1416)
    # env = gym.make(env_name, goal=goal) # multi-task learning

    # print(env.action_space, env.observation_space)

    n_steps_per_rollout = args.n_steps_per_rollout

    # --------------------------------------------------------------------------------------------------------------

    # START_ITER = 5000   #For 1M steps initialisation (Optimal hyperparameters)
    # # START_ITER = 25000  #For 5M steps initialisation (Just used for visualization right now)

    # SEARCH_INTERV = 1 # Since PPO make n_epochs=10 updates with each rollout, we can set this to 1 instead of 10

    # # NUM_ITERS = START_ITER + 100 # Just for testing
    # # NUM_ITERS = START_ITER + 20000 #5M steps (n_steps_per_rollout = 200)
    # NUM_ITERS = START_ITER + 7812 #5M steps (n_steps_per_rollout = 512)

    # N_EPOCHS = 10 # Since set to 10 updates per rollout

    # START_ITER = 1000000 // args.n_steps_per_rollout
    # SEARCH_INTERV = 1 # Since PPO make n_epochs=10 updates with each rollout, we can set this to 1 instead of 10
    # NUM_ITERS = 3000000 // args.n_steps_per_rollout
    # N_EPOCHS = args.n_epochs

    # ---------------------------------------------------------------------------------------------------------------

    exp = "PPO_gpu_rank_fqe" # For standard PPO training (single goal tasks)
    DIR = env_name + "/" + exp + "_" + str(get_latest_run_id('logs/'+env_name+"/", exp)+1)
    ckp_dir = f'logs/{DIR}/models'

    activation_fn_map = {
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
        'LeakyReLU': nn.LeakyReLU
    }

    if hasattr(args, 'use_policy_kwargs') and args.use_policy_kwargs:
        policy_kwargs = {
            "net_arch": [dict(pi=args.pi_layers, vf=args.vf_layers)],
            "activation_fn": activation_fn_map[args.activation_fn]
        }
        if hasattr(args, 'log_std_init'):
            policy_kwargs["log_std_init"] = args.log_std_init
        if hasattr(args, 'ortho_init'):
            policy_kwargs["ortho_init"] = args.ortho_init
    else:
        policy_kwargs = None

    if hasattr(args, 'use_normalize_kwargs') and args.use_normalize_kwargs:
        normalize_kwargs = {
            "norm_obs": args.norm_obs,
            "norm_reward": args.norm_reward
        }
    else:
        normalize_kwargs = None

    ppo_kwargs  = dict(
        policy=args.policy,
        env=env,
        verbose=args.verbose,
        seed=args.seed,
        n_steps=args.n_steps_per_rollout,
        gamma=args.gamma,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        ckp_dir=ckp_dir
    )

    if hasattr(args, 'max_grad_norm'):
        ppo_kwargs["max_grad_norm"] = args.max_grad_norm

    if hasattr(args, 'vf_coef'):
        ppo_kwargs["vf_coef"] = args.vf_coef

    if hasattr(args, 'clip_range'):
        ppo_kwargs["clip_range"] = args.clip_range

    if hasattr(args, 'learning_rate'):
        ppo_kwargs["learning_rate"] = args.learning_rate

    if hasattr(args, 'batch_size'): 
        ppo_kwargs["batch_size"] = args.batch_size

    if hasattr(args, 'normalize'):
        ppo_kwargs["normalize"] = args.normalize

    if hasattr(args, 'n_envs'):
        ppo_kwargs["n_envs"] = args.n_envs
    else:
        args.n_envs = 1

    if hasattr(args, 'sde_sample_freq'):
        ppo_kwargs["sde_sample_freq"] = args.sde_sample_freq

    if hasattr(args, 'ent_coef'):
        ppo_kwargs["ent_coef"] = args.ent_coef

    if policy_kwargs:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    if normalize_kwargs:
        ppo_kwargs["normalize_kwargs"] = normalize_kwargs

    model = PPO(**ppo_kwargs)

    # Replay-only semantics correction for FQE/OPE. This is installed before
    # every model.learn() call, including the optional initial 1M regeneration
    # block below. PPO rollout/training behavior is unchanged.
    install_fqe_replay_semantics_patch(model)

    START_ITER = 1000000 // (args.n_steps_per_rollout*args.n_envs)
    # START_ITER = 1
    SEARCH_INTERV = 1 # Make this 2 for n_epochs=5 and keep 1 for n_epochs=10

    if env_name == "Hopper-v5":
        SEARCH_INTERV = 2

    # NUM_ITERS = 3000000 // (args.n_steps_per_rollout*args.n_envs)
    # Replay sweet-spot study horizon. Compute NUM_ITERS only AFTER any
    # environment-specific SEARCH_INTERV override so the requested count is
    # truly the number of outer-loop observations for every environment.
    REPLAY_COVERAGE_NUM_OUTER_ITERS = int(
        os.environ.get("REPLAY_COVERAGE_NUM_OUTER_ITERS", "30")
    )
    if REPLAY_COVERAGE_NUM_OUTER_ITERS <= 0:
        raise ValueError("REPLAY_COVERAGE_NUM_OUTER_ITERS must be > 0.")
    NUM_ITERS = (
        START_ITER
        + SEARCH_INTERV * REPLAY_COVERAGE_NUM_OUTER_ITERS
    )
    # NUM_ITERS = 200000 // (args.n_steps_per_rollout*args.n_envs) # For FetchReach-v4
    N_EPOCHS = args.n_epochs

    # START_ITER = 1953
    # NUM_ITERS = 5858

    # ---------------------------------------------------------------------------------------------------------------

    # print("Starting Initial training")
    # os.makedirs(f'full_exp_on_ppo2/models/'+env_name, exist_ok=True)
    # os.makedirs(f'full_exp_on_ppo2/replay_buffers/'+env_name, exist_ok=True)

    # model.learn(total_timesteps=1000000, log_interval=50, tb_log_name=exp, init_call=True)
    # model.save("full_exp_on_ppo2/models/"+env_name+"/ppo_ant_1M"+'_'+str(args.seed))

    # print("Initial training done")

    # # Correct replay-buffer save format + corrected replay semantics for later FQE.
    # # IMPORTANT: buffers created before FQE_REPLAY_SEMANTICS_VERSION=2 must be
    # # regenerated once. Uncomment the initial-training block and this save block
    # # together so the saved PPO model and its replay buffer come from the same run.
    # print("Saving replay buffer for later use")
    # replay_buffer_path = (
    #     f'full_exp_on_ppo2/replay_buffers/{env_name}/'
    #     f'replay_buffer_{args.seed}.npz'
    # )
    # save_replay_buffer_npz(model, replay_buffer_path)

    # quit()

    # ----------------------------------------------------------------------------------------------------------------

    if START_ITER != 1 and env_name not in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:

        print("Loading Initial saved model")

        model.set_parameters(args.init_model_path+'_'+str(args.seed), device=args.device)

        print("Model loaded")

    else:
        print("Starting training from scratch")

    # -------------------------------------------------------------------------------------------------------------

    print("Loading replay buffer")

    replay_buffer_path = (
        f'full_exp_on_ppo2/replay_buffers/{env_name}/'
        f'replay_buffer_{args.seed}.npz'
    )
    load_replay_buffer_npz(model, replay_buffer_path)

    # ----------------------------------------------------------------------------------------------------------------

    vec_env = model.get_env()
    obs = vec_env.reset()

    print("Starting evaluation")

    normal_train = False
    use_ANN = False
    ANN_lib = "Annoy"

    # Keep the original online evaluation as the selector/oracle.
    online_eval = True

    # Rank-correlation study: additionally score the exact same candidates with FQE, but do
    # NOT use FQE to choose the next policy. The original online-selection trajectory remains
    # unchanged.
    rank_correlation_study = True
    if rank_correlation_study and not online_eval:
        raise ValueError(
            "rank_correlation_study requires online_eval=True because online returns are "
            "the ground truth and remain the selector during this study."
        )

    if STATE_OCCUPANCY_KNN_STUDY and not rank_correlation_study:
        raise ValueError(
            "STATE_OCCUPANCY_KNN_STUDY requires rank_correlation_study=True "
            "because it diagnoses FQE error against the existing online oracle."
        )

    saved_agents = False
    saved_iter = 4803
    model_already_learned = True

    distanceArray = []
    start_time = time.time()
    timeArray = []

    # Accumulates one row of rank-study metrics per outer iteration. This is analysis-only and
    # never participates in policy selection.
    rankStudyMetrics = []
    rankStudyCandidateRows = []

    # Objective-alignment study. These diagnostics never participate in policy
    # selection; the original undiscounted online return remains canonical.
    objectiveMismatchMetrics = []
    objectiveMismatchCandidateRows = []

    # Analysis-only replay coverage ablation. Never used for PPO/ESA selection.
    replayCoverageMetrics = []
    replayCoverageCandidateRows = []

    # Full-replay long-horizon FQE convergence diagnostic. Raw FQE only; never
    # used for PPO/ESA selection.
    fqeConvergenceMetrics = []
    fqeConvergenceCandidateRows = []

    # State-conditional kNN support/filter study. Analysis only; never used for
    # PPO/ESA selection.
    knnSupportMetrics = []
    knnSupportCandidateRows = []

    # State-occupancy kNN diagnostic. Uses copies of the SAME online candidate
    # states only after online returns are fixed; never used for replay, FQE
    # training, PPO/ESA selection, or best_idx.
    stateOccupancyMetrics = []
    stateOccupancyCandidateRows = []

    # Time-resolved state-occupancy diagnostic. Reuses the same captured
    # trajectories and the same 50k/100 FQE scores; analysis only.
    timeResolvedOccupancyMetrics = []
    timeResolvedOccupancyCandidateRows = []

    avg_checkpoint = False
    use_ptb = False

    parallel_evaluation = False

    if STATE_OCCUPANCY_KNN_STUDY and parallel_evaluation:
        raise NotImplementedError(
            "STATE_OCCUPANCY_KNN_STUDY currently requires the active "
            "non-parallel candidate-evaluation path so it can capture the "
            "exact same SB3 evaluate_policy trajectories without extra rollouts."
        )

    if exp == "PPO_baseline":
        # START_ITER = 1953
        # NUM_ITERS = 9765

        # For Pendulum-v1
        START_ITER = 976
        NUM_ITERS = 2930

    if saved_agents:
        # Find best agent index
        best_agent_index = np.load(f'logs/{env_name}/{exp}_{args.seed+1}/best_agent_{str(saved_iter-SEARCH_INTERV)}_{str(saved_iter)}.npy')
        print("Last Best agent index: ", best_agent_index[0])

        chosen_index = math.ceil((best_agent_index[0]+1) / 4)
        original_agent_index = 1 + 2*(chosen_index - 1)

        ckp = torch.load(f'logs/{env_name}/{exp}_{str(args.seed+1)}/models/agent{original_agent_index}.zip', map_location=device)
        print("Checkpoint loaded")

        load_state_dict(model, ckp)
        print("Model loaded")

        START_ITER = saved_iter

        # This study defines its horizon as a number of outer iterations.
        # Re-anchor the endpoint when resuming from saved_agents; otherwise
        # NUM_ITERS would still be tied to the pre-resume START_ITER and the
        # continuation loop could be shortened or empty.
        NUM_ITERS = (
            START_ITER
            + SEARCH_INTERV * REPLAY_COVERAGE_NUM_OUTER_ITERS
        )

    if not normal_train:
        for i in range(START_ITER, NUM_ITERS, SEARCH_INTERV):
            print(i)

            if saved_agents:
                if not model_already_learned:
                    model.learn(total_timesteps=SEARCH_INTERV*n_steps_per_rollout*vec_env.num_envs,
                                log_interval=1, 
                                tb_log_name=exp, 
                                reset_num_timesteps=True if i == START_ITER else False, 
                                first_iteration=True if i == START_ITER else False,
                                )

            else:
                model.learn(total_timesteps=SEARCH_INTERV*n_steps_per_rollout*vec_env.num_envs,
                            log_interval=1, 
                            tb_log_name=exp, 
                            reset_num_timesteps=True if i == START_ITER else False, 
                            first_iteration=True if i == START_ITER else False,
                            )

            # Diagnostics only: confirms that replay-only corrections are being
            # applied while PPO itself continues through the same learn() path.
            print_fqe_replay_semantics_stats(model)

            if not avg_checkpoint and not use_ptb:
                agents, distance = search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env, use_ANN, ANN_lib, saved_agents and model_already_learned, seed=args.seed)
                # agents, distance = neighbor_search_random_walk(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
                # agents, distance = random_search_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
                # agents, distance = random_search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
                # agents, distance = random_search_random_walk(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
                # agents, distance = search_guided_es_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env, saved_agents and model_already_learned)
                # agents, distance = search_vfs_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env, saved_agents and model_already_learned)
                distanceArray.append(distance)

            if saved_agents:
                saved_agents = False

            cum_rews = []
            # Diagnostic-only gamma-discounted returns from the SAME online
            # trajectories used to populate cum_rews. Never used for selection.
            cum_discounted_rews = []
            # Optional read-only copies of the SAME online candidate states.
            # One list entry per candidate; each entry contains the 3 completed
            # evaluation episodes. These traces never enter PPO replay/FQE.
            candidate_occupancy_trajectories = []
            cum_success = []
            best_agent_index = []
            advantage_rew = []
            # q_losses = []

            # -----------------------------------------------------------------------------------

            if avg_checkpoint:
                # Average the last checkpoints
                checkpoint_paths = [f'logs/{DIR}/models/agent{j}.zip' for j in range(1, 11)]
                avg_policy_vec = average_checkpoints(checkpoint_paths)
                avg_policy_vec = avg_policy_vec.reshape(1, -1)
                print("Average policy vector shape: ", avg_policy_vec.shape)
                agents = dump_weights(model.policy.state_dict(), avg_policy_vec)

                model.policy.load_state_dict(agents[0])
                model.policy.to(device)

                # Online evaluation
                if hasattr(args, 'n_envs') and args.n_envs > 1:
                    # Create a list of environment functions
                    dummy_env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=i) for i in range(args.n_envs)]
                    dummy_env = SubprocVecEnv(dummy_env_fns)
                else:
                    dummy_env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5
                    dummy_env.reset(seed=args.seed)

                returns_trains = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True)[0]
                print(f'avg return on 3 trajectories of agent: {returns_trains}')
                cum_rews.append(returns_trains)
                close_env_safely(dummy_env)

                os.makedirs(f'logs/{DIR}', exist_ok=True)
                np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents_to_cpu(agents))
                np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
                timeArray.append(time.time() - start_time)

                load_state_dict(model, agents[0])

                continue

            # -----------------------------------------------------------------------------------

            if use_ptb:
                print("Using PBT to search policies")

                checkpoint_paths = [f'logs/{DIR}/models/agent{j}.zip' for j in range(1, 11)]

                policies = []
                for path in checkpoint_paths:
                    policy_vec = []
                    ckp = torch.load(path, map_location='cpu')
                    ckp_layers = ckp.keys()

                    for layer in ckp_layers:
                        if 'value_net' not in layer:
                            policy_vec.append(ckp[layer].detach().numpy().reshape(-1))

                    policy_vec = np.concatenate(policy_vec)
                    policies.append(policy_vec)

                rewards = []
                for _, vec in enumerate(policies):
                    policy_vec = vec.reshape(1, -1)
                    a = dump_weights(model.policy.state_dict(), policy_vec)
                    model.policy.load_state_dict(a[0])
                    model.policy.to(device)

                    # Online evaluation
                    if hasattr(args, 'n_envs') and args.n_envs > 1:
                        # Create a list of environment functions
                        dummy_env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=z) for z in range(args.n_envs)]
                        dummy_env = SubprocVecEnv(dummy_env_fns)
                    else:
                        dummy_env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5
                        dummy_env.reset(seed=args.seed)

                    returns_trains = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True)[0]
                    rewards.append(returns_trains)
                    close_env_safely(dummy_env)

                rewards = np.array(rewards)
                top_indices = rewards.argsort()[-5:]
                bottom_indices = rewards.argsort()[:len(checkpoint_paths) - 5]

                new_policies = []
                for j in top_indices:
                    new_policies.append(policies[j])  # keep top agents

                for j in bottom_indices:
                    parent = np.random.choice(top_indices)
                    noise = np.random.normal(scale=0.02, size=policies[parent].shape)
                    mutated = policies[parent] + noise
                    new_policies.append(mutated)

                new_policies = np.array(new_policies)

                agents = dump_weights(model.policy.state_dict(), new_policies)

            # -----------------------------------------------------------------------------------

            # For the rank-correlation study, freeze the PPO replay-buffer data once per outer
            # iteration, after the candidate set has been constructed. Every candidate FQE fit
            # below receives this same dataset. Candidate online-evaluation trajectories must
            # never enter this buffer.
            if rank_correlation_study:
                fqe_dataset = build_fqe_dataset(model)

                # Freeze corrected native transition views at the same point in time,
                # BEFORE online candidate evaluation. Candidate rollouts therefore cannot
                # enter any replay-coverage window.
                replay_coverage_data = None
                if FQE_BACKEND == "native_batched":
                    if REPLAY_COVERAGE_ABLATION:
                        replay_coverage_data = OrderedDict()
                        for coverage_window in REPLAY_COVERAGE_WINDOWS:
                            coverage_label = (
                                "full" if coverage_window is None
                                else str(int(coverage_window))
                            )
                            replay_coverage_data[coverage_label] = (
                                build_native_fqe_replay_data(
                                    model,
                                    max_transitions=coverage_window,
                                    finite_horizon_steps=(
                                        fqe_finite_horizon_steps
                                    ),
                                )
                            )

                        # Replay-coverage ablation must vary only the FQE
                        # TRAINING replay. Use one common frozen start-state
                        # reference set for scoring every window; otherwise the
                        # experiment changes both replay coverage and the s0
                        # distribution used by E[Q(s0, pi(s0))].
                        coverage_reference_initial_observations = np.array(
                            replay_coverage_data["full"]["initial_observations"],
                            dtype=np.float32,
                            copy=True,
                        )
                        coverage_reference_initial_timesteps = np.zeros(
                            (
                                len(coverage_reference_initial_observations),
                                1,
                            ),
                            dtype=np.float32,
                        )

                        for coverage_data in replay_coverage_data.values():
                            coverage_data["training_initial_states"] = int(
                                len(coverage_data["initial_observations"])
                            )
                            coverage_data["initial_observations"] = (
                                coverage_reference_initial_observations.copy()
                            )
                            coverage_data["initial_timesteps"] = (
                                coverage_reference_initial_timesteps.copy()
                            )
                            coverage_data["score_initial_states"] = int(
                                len(coverage_reference_initial_observations)
                            )

                        print(
                            "Replay coverage scoring reference: "
                            f"{len(coverage_reference_initial_observations)} "
                            "common full-buffer initial states."
                        )

                        # Preserve original/canonical behavior: the existing rank study
                        # remains defined by the full corrected replay buffer.
                        native_fqe_data = replay_coverage_data["full"]
                    else:
                        native_fqe_data = build_native_fqe_replay_data(
                            model,
                            finite_horizon_steps=fqe_finite_horizon_steps,
                        )

                    # The behavioral trust-region penalty uses one COMMON full
                    # corrected replay reference for all coverage windows. This
                    # keeps the coverage ablation scientifically clean: only FQE
                    # training coverage changes across windows, not the support
                    # metric itself.
                    support_reference_data = native_fqe_data
                else:
                    native_fqe_data = None
                    # d3rlpy still receives the same explicit behavior-support
                    # penalty, built from the corrected replay semantics.
                    support_reference_data = build_native_fqe_replay_data(
                        model,
                        finite_horizon_steps=fqe_finite_horizon_steps,
                    )

                replay_buffer_before_candidate_eval = replay_buffer_signature(model.replay_buffer)
                print(
                    "Rank-correlation study enabled: frozen one FQE dataset for all "
                    f"{len(agents)} candidates in iteration {i}. "
                    f"FQE backend: {FQE_BACKEND}"
                )
            else:
                fqe_dataset = None
                native_fqe_data = None
                replay_coverage_data = None
                support_reference_data = None
                replay_buffer_before_candidate_eval = None

            # Non-parallel evaluation (Commented out)
            if not parallel_evaluation:
                for j, a in enumerate(agents):
                    model.policy.load_state_dict(a)
                    model.policy.to(device)

                    # Online evaluation
                    if hasattr(args, 'n_envs') and args.n_envs > 1:
                        # Create a list of environment functions
                        dummy_env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=i) for i in range(args.n_envs)]
                        dummy_env = SubprocVecEnv(dummy_env_fns)
                    else:
                        dummy_env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5

                        if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                            dummy_env = FlattenObservation(dummy_env)

                        dummy_env.reset(seed=args.seed)

                    # The state-occupancy diagnostic is collected by a read-only
                    # callback over the SAME evaluate_policy call. No additional
                    # online rollout is introduced.
                    need_readonly_online_callback = (
                        rank_correlation_study
                        and (
                            OBJECTIVE_MISMATCH_STUDY
                            or STATE_OCCUPANCY_KNN_STUDY
                        )
                    )

                    if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                        if need_readonly_online_callback:
                            diagnostic_result = (
                                evaluate_policy_with_discounted_return(
                                    model,
                                    dummy_env,
                                    n_eval_episodes=3,
                                    gamma=model.gamma,
                                    deterministic=True,
                                    capture_trajectory=(
                                        STATE_OCCUPANCY_KNN_STUDY
                                    ),
                                    return_success_rate=True,
                                )
                            )
                            if STATE_OCCUPANCY_KNN_STUDY:
                                (
                                    eval_result,
                                    discounted_return,
                                    _,
                                    trajectory_episodes,
                                ) = diagnostic_result
                                candidate_occupancy_trajectories.append(
                                    trajectory_episodes
                                )
                            else:
                                (
                                    eval_result,
                                    discounted_return,
                                    _,
                                ) = diagnostic_result

                            mean_rew, std_rew, success = eval_result
                            if OBJECTIVE_MISMATCH_STUDY:
                                cum_discounted_rews.append(
                                    discounted_return
                                )
                        else:
                            mean_rew, std_rew, success = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True, return_success_rate=True)
                        print(f'avg 3 return on policy: {mean_rew}, Success rate: {success:.2f}')
                        if OBJECTIVE_MISMATCH_STUDY and rank_correlation_study:
                            print(
                                f'avg gamma-discounted return on same 3 trajectories: '
                                f'{cum_discounted_rews[-1]}'
                            )
                        cum_rews.append(mean_rew)
                        cum_success.append(success)
                    else:
                        if need_readonly_online_callback:
                            diagnostic_result = (
                                evaluate_policy_with_discounted_return(
                                    model,
                                    dummy_env,
                                    n_eval_episodes=3,
                                    gamma=model.gamma,
                                    deterministic=True,
                                    capture_trajectory=(
                                        STATE_OCCUPANCY_KNN_STUDY
                                    ),
                                )
                            )
                            if STATE_OCCUPANCY_KNN_STUDY:
                                (
                                    eval_result,
                                    discounted_return,
                                    _,
                                    trajectory_episodes,
                                ) = diagnostic_result
                                candidate_occupancy_trajectories.append(
                                    trajectory_episodes
                                )
                            else:
                                (
                                    eval_result,
                                    discounted_return,
                                    _,
                                ) = diagnostic_result

                            returns_trains = eval_result[0]
                            if OBJECTIVE_MISMATCH_STUDY:
                                cum_discounted_rews.append(
                                    discounted_return
                                )
                        else:
                            returns_trains = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True)[0]
                        print(f'avg return on 3 trajectories of agent{j}: {returns_trains}')
                        if OBJECTIVE_MISMATCH_STUDY and rank_correlation_study:
                            print(
                                f'avg gamma-discounted return on same 3 trajectories '
                                f'of agent{j}: {cum_discounted_rews[-1]}'
                            )
                        cum_rews.append(returns_trains)

                    close_env_safely(dummy_env)

                    # Q-function / FQE evaluation.
                    #
                    # In rank_correlation_study mode, defer FQE until ALL
                    # candidates have completed the exact same online evaluation.
                    # The previous per-candidate FQE call preserved/restored RNG
                    # and never mutated the replay buffer, so moving all shadow
                    # FQE work after the online loop leaves the online oracle
                    # trajectory unchanged while enabling one batched GPU fit.
                    #
                    # Keep the legacy offline-selection path unchanged.
                    if not rank_correlation_study and not online_eval:
                        fqe_exp_name = f"{'-'.join(DIR.split('/'))}"
                        init_est = d3rl_evaluation(model, fqe_exp_name)
                        if init_est is None:
                            raise RuntimeError(
                                f"FQE failed for iteration {i}, agent {j}."
                            )
                        init_est = float(np.asarray(init_est).reshape(-1)[0])
                        advantage_rew.append(init_est)

            # Parallel evaluation
            else:
                if OBJECTIVE_MISMATCH_STUDY and rank_correlation_study:
                    cum_rews, cum_discounted_rews = parallel_evaluate(
                        agents=agents,
                        env_name=env_name,
                        n_eval_episodes=3,
                        seed=args.seed,
                        gamma=model.gamma,
                        return_discounted=True,
                    )
                else:
                    cum_rews = parallel_evaluate(
                        agents=agents,
                        env_name=env_name,
                        n_eval_episodes=3,
                        seed=args.seed
                    )

                # parallel_evaluate performs only the online ground-truth
                # rollouts. The discounted diagnostic, when enabled, is
                # accumulated inside those SAME rollouts. Rank-study FQE is
                # intentionally deferred to the common batched block below.

            # --------------------------------------------------------------
            # ANALYSIS-ONLY STATE-OCCUPANCY kNN DIAGNOSTIC
            # --------------------------------------------------------------
            # Uses copies of states from the SAME completed online evaluation
            # trajectories. These states are not inserted into replay and are
            # not available to FQE training or the online selector.
            state_occupancy_diagnostics = None
            if STATE_OCCUPANCY_KNN_STUDY:
                if len(candidate_occupancy_trajectories) != len(agents):
                    raise RuntimeError(
                        "State-occupancy trajectory count mismatch: "
                        f"{len(candidate_occupancy_trajectories)} traces for "
                        f"{len(agents)} candidate policies."
                    )
                state_occupancy_diagnostics = (
                    compute_state_occupancy_knn_diagnostics(
                        candidate_trajectory_episodes=(
                            candidate_occupancy_trajectories
                        ),
                        replay_data=support_reference_data,
                    )
                )

                print("---------------------------------")
                print("STATE-OCCUPANCY kNN DIAGNOSTIC")
                print(
                    f"reference={state_occupancy_diagnostics['reference_count']}, "
                    f"replay_queries="
                    f"{state_occupancy_diagnostics['replay_query_count']}, "
                    f"k={state_occupancy_diagnostics['k']}, "
                    f"time_aware="
                    f"{state_occupancy_diagnostics['include_time']}, "
                    f"behavior_percentile="
                    f"{state_occupancy_diagnostics['behavior_percentile']:.1f}"
                )
                print(
                    "Replay leave-one-out kNN-radius threshold: "
                    f"{state_occupancy_diagnostics['behavior_threshold_knn_radius']:.6f}"
                )
                print(
                    "Candidate occupancy OOD fractions: "
                    + np.array2string(
                        state_occupancy_diagnostics[
                            'candidate_ood_fraction'
                        ],
                        precision=4,
                        separator=", ",
                        max_line_width=160,
                    )
                )
                print(
                    "Candidate mean kNN radii: "
                    + np.array2string(
                        state_occupancy_diagnostics[
                            'candidate_mean_knn_radius'
                        ],
                        precision=6,
                        separator=", ",
                        max_line_width=160,
                    )
                )

            # Rank-correlation OPE is analysis-only and is evaluated after all
            # online candidate returns are already fixed. This preserves the
            # original online oracle. The new ranking score is:
            #   ordinary FQE value - lambda * replay action divergence.
            if rank_correlation_study:
                # Compute the behavioral trust-region term ONCE per outer
                # iteration from the common frozen full-buffer support reference.
                # Candidate online rollouts have already completed, but they are
                # not stored in this replay buffer, and the leakage invariant below
                # verifies that fact.
                support_action_divergence = (
                    compute_behavior_action_divergence_preserving_rng(
                        model,
                        agents,
                        support_reference_data,
                    )
                )
                support_reference_label = str(
                    support_reference_data.get("coverage_label", "full")
                )
                support_reference_transitions = int(
                    support_reference_data.get(
                        "actual_transitions",
                        len(support_reference_data["observations"]),
                    )
                )

                knn_support_diagnostics = None
                if KNN_SUPPORT_STUDY:
                    knn_support_diagnostics = (
                        compute_state_conditional_knn_support_preserving_rng(
                            model, agents, support_reference_data
                        )
                    )
                    print("---------------------------------")
                    print("STATE-CONDITIONAL kNN SUPPORT ESTIMATOR")
                    print(
                        f"reference={knn_support_diagnostics['reference_count']}, "
                        f"queries={knn_support_diagnostics['query_count']}, "
                        f"k={knn_support_diagnostics['k']}, "
                        f"behavior_percentile="
                        f"{knn_support_diagnostics['behavior_percentile']:.1f}"
                    )
                    print(
                        "Behavior leave-one-out local-action threshold "
                        "(squared L2): "
                        f"{knn_support_diagnostics['behavior_threshold_sq_l2']:.6f}"
                    )
                    print(
                        "Candidate unsupported-state fractions: "
                        + np.array2string(
                            knn_support_diagnostics[
                                'candidate_unsupported_fraction'
                            ],
                            precision=4,
                            separator=", ",
                            max_line_width=160,
                        )
                    )
                    print(
                        "Candidate mean nearest-local-action squared L2: "
                        + np.array2string(
                            knn_support_diagnostics['candidate_mean_sq_l2'],
                            precision=6,
                            separator=", ",
                            max_line_width=160,
                        )
                    )

                print("---------------------------------")
                print("BEHAVIOR SUPPORT PENALTY")
                print(
                    f"lambda={FQE_SUPPORT_PENALTY_LAMBDA:g}, "
                    f"reference={support_reference_label}, "
                    f"transitions={support_reference_transitions}"
                )
                print(
                    "Mean squared-L2 action divergence per candidate: "
                    + np.array2string(
                        support_action_divergence,
                        precision=6,
                        separator=", ",
                        max_line_width=160,
                    )
                )

                if FQE_BACKEND == "native_batched":
                    replay_coverage_scores = None
                    canonical_base_fqe_scores = None

                    if REPLAY_COVERAGE_ABLATION:
                        replay_coverage_scores = OrderedDict()

                        print("---------------------------------")
                        print("REPLAY COVERAGE ABLATION")
                        print(
                            "Windows: "
                            + ", ".join(replay_coverage_data.keys())
                        )

                        for coverage_label, coverage_data in replay_coverage_data.items():
                            print("---------------------------------")
                            print(
                                "Running FQE replay coverage window: "
                                f"{coverage_label} "
                                f"(actual complete transitions="
                                f"{coverage_data['actual_transitions']}, "
                                f"episodes={coverage_data['n_episodes']})"
                            )
                            base_fqe_scores = native_batched_fqe_preserving_rng(
                                model,
                                agents,
                                fqe_dataset,
                                native_data=coverage_data,
                            )
                            if coverage_label == "full":
                                canonical_base_fqe_scores = base_fqe_scores
                            replay_coverage_scores[coverage_label] = (
                                build_support_penalized_scores(
                                    base_fqe_scores=base_fqe_scores,
                                    action_divergence=support_action_divergence,
                                    support_reference_label=support_reference_label,
                                    support_reference_transitions=(
                                        support_reference_transitions
                                    ),
                                )
                            )

                        # CRITICAL: preserve original/canonical rank-study semantics.
                        # Downstream rank metrics still use the FULL replay FQE
                        # estimate, now augmented only by the explicit support term.
                        advantage_rew = replay_coverage_scores["full"]
                    else:
                        base_fqe_scores = native_batched_fqe_preserving_rng(
                            model,
                            agents,
                            fqe_dataset,
                            native_data=native_fqe_data,
                        )
                        canonical_base_fqe_scores = base_fqe_scores
                        advantage_rew = build_support_penalized_scores(
                            base_fqe_scores=base_fqe_scores,
                            action_divergence=support_action_divergence,
                            support_reference_label=support_reference_label,
                            support_reference_transitions=(
                                support_reference_transitions
                            ),
                        )
                    knn_support_fqe_scores = None

                    # ----------------------------------------------------------
                    # LONG-HORIZON FQE CONVERGENCE / PROPAGATION STUDY
                    # ----------------------------------------------------------
                    # Compare raw FQE on the SAME full replay data while varying
                    # only optimization steps and target-network refresh interval.
                    # The canonical 10k/100 fit is reused when present, avoiding a
                    # redundant fit. Support penalties are deliberately excluded.
                    if FQE_CONVERGENCE_STUDY:
                        if canonical_base_fqe_scores is None:
                            raise RuntimeError(
                                "FQE convergence study could not locate the canonical "
                                "full-replay base FQE fit."
                            )

                        print("---------------------------------")
                        print("FQE LONG-HORIZON CONVERGENCE STUDY")
                        print(
                            "Configs: "
                            + ", ".join(
                                f"{steps} steps / target {target_interval}"
                                for steps, target_interval in FQE_CONVERGENCE_CONFIGS
                            )
                        )

                        for conv_steps, conv_target_interval in FQE_CONVERGENCE_CONFIGS:
                            if (
                                int(conv_steps) == int(FQE_N_STEPS)
                                and int(conv_target_interval)
                                == int(NATIVE_FQE_TARGET_UPDATE_INTERVAL)
                            ):
                                conv_scores = canonical_base_fqe_scores
                                reused_canonical = True
                            else:
                                conv_scores = native_batched_fqe_preserving_rng(
                                    model,
                                    agents,
                                    fqe_dataset,
                                    native_data=native_fqe_data,
                                    n_steps=int(conv_steps),
                                    target_update_interval=int(conv_target_interval),
                                )
                                reused_canonical = False

                            if (
                                int(conv_steps) == int(KNN_SUPPORT_FQE_N_STEPS)
                                and int(conv_target_interval)
                                == int(KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL)
                            ):
                                knn_support_fqe_scores = conv_scores

                            conv_raw_q = np.asarray(
                                getattr(conv_scores, "mean_q", conv_scores),
                                dtype=np.float64,
                            )
                            conv_metrics = compute_fqe_convergence_metrics(
                                online_scores=np.asarray(cum_rews, dtype=np.float64),
                                raw_fqe_scores=conv_raw_q,
                                iteration=i,
                                n_steps=int(conv_steps),
                                target_update_interval=int(conv_target_interval),
                                score_metadata=conv_scores,
                            )
                            conv_metrics["reused_canonical_fit"] = bool(
                                reused_canonical
                            )
                            fqeConvergenceMetrics.append(conv_metrics)

                            for candidate_idx in range(len(conv_raw_q)):
                                fqeConvergenceCandidateRows.append({
                                    "iteration": int(i),
                                    "candidate": int(candidate_idx),
                                    "config": conv_metrics["config"],
                                    "fqe_n_steps": int(conv_steps),
                                    "fqe_target_update_interval": int(
                                        conv_target_interval
                                    ),
                                    "fqe_target_updates": int(
                                        conv_metrics["fqe_target_updates"]
                                    ),
                                    "fqe_objective": conv_metrics["fqe_objective"],
                                    "fqe_gamma": conv_metrics["fqe_gamma"],
                                    "finite_horizon_steps": conv_metrics[
                                        "finite_horizon_steps"
                                    ],
                                    "time_conditioned": conv_metrics[
                                        "time_conditioned"
                                    ],
                                    "fqe_mean_q": float(conv_raw_q[candidate_idx]),
                                    "final_fqe_loss": float(
                                        np.asarray(
                                            getattr(
                                                conv_scores,
                                                "final_loss_per_candidate",
                                                np.full(len(conv_raw_q), np.nan),
                                            ),
                                            dtype=np.float64,
                                        )[candidate_idx]
                                    ),
                                    "online": float(cum_rews[candidate_idx]),
                                })

                            print(
                                f"{conv_metrics['config']} | "
                                f"target_updates={conv_metrics['fqe_target_updates']} | "
                                f"meanQ={conv_metrics['mean_fqe_q']:.3f} | "
                                f"Qrange={conv_metrics['fqe_q_range']:.3f} | "
                                f"final_loss={conv_metrics['mean_final_fqe_loss']:.4f} | "
                                f"Pearson={conv_metrics['pearson']:+.4f} | "
                                f"Spearman={conv_metrics['spearman']:+.4f} | "
                                f"Kendall={conv_metrics['kendall']:+.4f} | "
                                f"top1={int(conv_metrics['top1_agreement'])} | "
                                f"regret={conv_metrics['selection_regret']:.4f}"
                            )

                    # ----------------------------------------------------------
                    # STATE-CONDITIONAL kNN SUPPORT FILTER STUDY
                    # ----------------------------------------------------------
                    if KNN_SUPPORT_STUDY:
                        if knn_support_diagnostics is None:
                            raise RuntimeError(
                                "kNN support study is missing support diagnostics."
                            )

                        # Reuse the canonical fit if the user has already made
                        # it the requested 50k/100 support-FQE configuration.
                        if (
                            knn_support_fqe_scores is None
                            and int(getattr(
                                canonical_base_fqe_scores,
                                'fqe_n_steps',
                                FQE_N_STEPS,
                            )) == int(KNN_SUPPORT_FQE_N_STEPS)
                            and int(getattr(
                                canonical_base_fqe_scores,
                                'fqe_target_update_interval',
                                NATIVE_FQE_TARGET_UPDATE_INTERVAL,
                            )) == int(KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL)
                        ):
                            knn_support_fqe_scores = canonical_base_fqe_scores

                        if knn_support_fqe_scores is None:
                            print("---------------------------------")
                            print(
                                "Fitting primary FQE for kNN support filter: "
                                f"{KNN_SUPPORT_FQE_N_STEPS} steps / target "
                                f"{KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL}"
                            )
                            knn_support_fqe_scores = (
                                native_batched_fqe_preserving_rng(
                                    model,
                                    agents,
                                    fqe_dataset,
                                    native_data=native_fqe_data,
                                    n_steps=KNN_SUPPORT_FQE_N_STEPS,
                                    target_update_interval=(
                                        KNN_SUPPORT_FQE_TARGET_UPDATE_INTERVAL
                                    ),
                                )
                            )

                        knn_raw_fqe = np.asarray(
                            getattr(
                                knn_support_fqe_scores,
                                'mean_q',
                                knn_support_fqe_scores,
                            ),
                            dtype=np.float64,
                        )
                        knn_metrics, knn_candidate_details = (
                            compute_knn_support_filter_metrics(
                                online_scores=np.asarray(
                                    cum_rews, dtype=np.float64
                                ),
                                raw_fqe_scores=knn_raw_fqe,
                                support_diagnostics=knn_support_diagnostics,
                                iteration=i,
                                score_metadata=knn_support_fqe_scores,
                            )
                        )
                        knnSupportMetrics.append(knn_metrics)

                        for candidate_idx in range(len(knn_raw_fqe)):
                            knnSupportCandidateRows.append({
                                'iteration': int(i),
                                'candidate': int(candidate_idx),
                                'online': float(cum_rews[candidate_idx]),
                                'fqe_mean_q': float(
                                    knn_raw_fqe[candidate_idx]
                                ),
                                'fqe_n_steps': int(
                                    knn_metrics['fqe_n_steps']
                                ),
                                'fqe_target_update_interval': int(
                                    knn_metrics[
                                        'fqe_target_update_interval'
                                    ]
                                ),
                                'knn_k': int(
                                    knn_support_diagnostics['k']
                                ),
                                'knn_query_states': int(
                                    knn_support_diagnostics['query_count']
                                ),
                                'behavior_percentile': float(
                                    knn_support_diagnostics[
                                        'behavior_percentile'
                                    ]
                                ),
                                'behavior_threshold_sq_l2': float(
                                    knn_support_diagnostics[
                                        'behavior_threshold_sq_l2'
                                    ]
                                ),
                                'knn_mean_sq_l2': float(
                                    knn_support_diagnostics[
                                        'candidate_mean_sq_l2'
                                    ][candidate_idx]
                                ),
                                'knn_median_sq_l2': float(
                                    knn_support_diagnostics[
                                        'candidate_median_sq_l2'
                                    ][candidate_idx]
                                ),
                                'knn_p95_sq_l2': float(
                                    knn_support_diagnostics[
                                        'candidate_p95_sq_l2'
                                    ][candidate_idx]
                                ),
                                'knn_mean_excess_sq_l2': float(
                                    knn_support_diagnostics[
                                        'candidate_mean_excess_sq_l2'
                                    ][candidate_idx]
                                ),
                                'knn_mean_ratio_to_behavior': float(
                                    knn_support_diagnostics[
                                        'candidate_mean_ratio_to_behavior'
                                    ][candidate_idx]
                                ),
                                'knn_unsupported_fraction': float(
                                    knn_support_diagnostics[
                                        'candidate_unsupported_fraction'
                                    ][candidate_idx]
                                ),
                                'absolute_filter_pass': bool(
                                    knn_candidate_details[
                                        'threshold_pass'
                                    ][candidate_idx]
                                ),
                                'effective_filter_keep': bool(
                                    knn_candidate_details[
                                        'effective_keep'
                                    ][candidate_idx]
                                ),
                                'support_rank': int(
                                    knn_candidate_details['support_rank'][
                                        candidate_idx
                                    ]
                                ),
                                'raw_fqe_rank': int(
                                    knn_candidate_details['raw_fqe_rank'][
                                        candidate_idx
                                    ]
                                ),
                                'filtered_fqe_rank': int(
                                    knn_candidate_details[
                                        'filtered_fqe_rank'
                                    ][candidate_idx]
                                ),
                            })

                        print("---------------------------------")
                        print("STATE-CONDITIONAL kNN SUPPORT FILTER")
                        print(
                            f"absolute_pass="
                            f"{knn_metrics['threshold_keep_count']}/"
                            f"{len(knn_raw_fqe)}, "
                            f"effective_keep="
                            f"{knn_metrics['effective_keep_count']}/"
                            f"{len(knn_raw_fqe)}, "
                            f"fallback_added="
                            f"{knn_metrics['fallback_fill_count']}"
                        )
                        print(
                            f"raw 50k/100 FQE idx="
                            f"{knn_metrics['raw_fqe_idx']} | "
                            f"regret="
                            f"{knn_metrics['raw_fqe_selection_regret']:.4f}"
                        )
                        print(
                            f"filtered FQE idx="
                            f"{knn_metrics['filtered_fqe_idx']} | "
                            f"regret="
                            f"{knn_metrics['filtered_fqe_selection_regret']:.4f} | "
                            f"oracle_survives_filter="
                            f"{knn_metrics['oracle_survives_effective_filter']}"
                        )
                        for requested_k in HYBRID_TOPK_VALUES:
                            print(
                                f"  k={requested_k}: "
                                f"raw Recall="
                                f"{int(knn_metrics[f'raw_oracle_recall_at_{requested_k}'])}, "
                                f"raw HReg="
                                f"{knn_metrics[f'raw_hybrid_regret_at_{requested_k}']:.4f} | "
                                f"filtered effective_k="
                                f"{knn_metrics[f'filtered_effective_k_at_{requested_k}']}, "
                                f"Recall="
                                f"{int(knn_metrics[f'filtered_oracle_recall_at_{requested_k}'])}, "
                                f"HReg="
                                f"{knn_metrics[f'filtered_hybrid_regret_at_{requested_k}']:.4f}"
                            )

                    # ----------------------------------------------------------
                    # STATE-OCCUPANCY kNN / FQE-ERROR DIAGNOSTIC
                    # ----------------------------------------------------------
                    if STATE_OCCUPANCY_KNN_STUDY:
                        if state_occupancy_diagnostics is None:
                            raise RuntimeError(
                                "State-occupancy study is missing trajectory "
                                "diagnostics."
                            )

                        occupancy_fqe_scores = None

                        # Reuse any already-computed 50k/100 fit from the kNN
                        # support/convergence diagnostics when it exactly matches
                        # the occupancy study's requested estimator.
                        if (
                            knn_support_fqe_scores is not None
                            and int(getattr(
                                knn_support_fqe_scores,
                                'fqe_n_steps',
                                -1,
                            )) == int(STATE_OCCUPANCY_FQE_N_STEPS)
                            and int(getattr(
                                knn_support_fqe_scores,
                                'fqe_target_update_interval',
                                -1,
                            )) == int(
                                STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL
                            )
                        ):
                            occupancy_fqe_scores = knn_support_fqe_scores

                        if (
                            occupancy_fqe_scores is None
                            and canonical_base_fqe_scores is not None
                            and int(getattr(
                                canonical_base_fqe_scores,
                                'fqe_n_steps',
                                FQE_N_STEPS,
                            )) == int(STATE_OCCUPANCY_FQE_N_STEPS)
                            and int(getattr(
                                canonical_base_fqe_scores,
                                'fqe_target_update_interval',
                                NATIVE_FQE_TARGET_UPDATE_INTERVAL,
                            )) == int(
                                STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL
                            )
                        ):
                            occupancy_fqe_scores = canonical_base_fqe_scores

                        if occupancy_fqe_scores is None:
                            print("---------------------------------")
                            print(
                                "Fitting primary FQE for state-occupancy "
                                "diagnostic: "
                                f"{STATE_OCCUPANCY_FQE_N_STEPS} steps / "
                                f"target "
                                f"{STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL}"
                            )
                            occupancy_fqe_scores = (
                                native_batched_fqe_preserving_rng(
                                    model,
                                    agents,
                                    fqe_dataset,
                                    native_data=native_fqe_data,
                                    n_steps=STATE_OCCUPANCY_FQE_N_STEPS,
                                    target_update_interval=(
                                        STATE_OCCUPANCY_FQE_TARGET_UPDATE_INTERVAL
                                    ),
                                )
                            )

                        occupancy_raw_fqe = np.asarray(
                            getattr(
                                occupancy_fqe_scores,
                                'mean_q',
                                occupancy_fqe_scores,
                            ),
                            dtype=np.float64,
                        )
                        (
                            occupancy_metrics,
                            occupancy_candidate_details,
                        ) = compute_state_occupancy_fqe_metrics(
                            online_scores=np.asarray(
                                cum_rews, dtype=np.float64
                            ),
                            raw_fqe_scores=occupancy_raw_fqe,
                            occupancy_diagnostics=(
                                state_occupancy_diagnostics
                            ),
                            iteration=i,
                            score_metadata=occupancy_fqe_scores,
                        )
                        stateOccupancyMetrics.append(occupancy_metrics)

                        for candidate_idx in range(
                            len(occupancy_raw_fqe)
                        ):
                            stateOccupancyCandidateRows.append({
                                'iteration': int(i),
                                'candidate': int(candidate_idx),
                                'online': float(cum_rews[candidate_idx]),
                                'fqe_mean_q': float(
                                    occupancy_raw_fqe[candidate_idx]
                                ),
                                'fqe_n_steps': int(
                                    occupancy_metrics['fqe_n_steps']
                                ),
                                'fqe_target_update_interval': int(
                                    occupancy_metrics[
                                        'fqe_target_update_interval'
                                    ]
                                ),
                                'knn_k': int(
                                    state_occupancy_diagnostics['k']
                                ),
                                'time_aware': bool(
                                    state_occupancy_diagnostics[
                                        'include_time'
                                    ]
                                ),
                                'behavior_percentile': float(
                                    state_occupancy_diagnostics[
                                        'behavior_percentile'
                                    ]
                                ),
                                'behavior_threshold_knn_radius': float(
                                    state_occupancy_diagnostics[
                                        'behavior_threshold_knn_radius'
                                    ]
                                ),
                                'trajectory_total_states': int(
                                    state_occupancy_diagnostics[
                                        'candidate_total_states'
                                    ][candidate_idx]
                                ),
                                'trajectory_query_states': int(
                                    state_occupancy_diagnostics[
                                        'candidate_query_states'
                                    ][candidate_idx]
                                ),
                                'occupancy_mean_knn_radius': float(
                                    state_occupancy_diagnostics[
                                        'candidate_mean_knn_radius'
                                    ][candidate_idx]
                                ),
                                'occupancy_median_knn_radius': float(
                                    state_occupancy_diagnostics[
                                        'candidate_median_knn_radius'
                                    ][candidate_idx]
                                ),
                                'occupancy_p95_knn_radius': float(
                                    state_occupancy_diagnostics[
                                        'candidate_p95_knn_radius'
                                    ][candidate_idx]
                                ),
                                'occupancy_max_knn_radius': float(
                                    state_occupancy_diagnostics[
                                        'candidate_max_knn_radius'
                                    ][candidate_idx]
                                ),
                                'occupancy_mean_1nn_distance': float(
                                    state_occupancy_diagnostics[
                                        'candidate_mean_1nn_distance'
                                    ][candidate_idx]
                                ),
                                'occupancy_p95_1nn_distance': float(
                                    state_occupancy_diagnostics[
                                        'candidate_p95_1nn_distance'
                                    ][candidate_idx]
                                ),
                                'occupancy_ood_fraction': float(
                                    state_occupancy_diagnostics[
                                        'candidate_ood_fraction'
                                    ][candidate_idx]
                                ),
                                'occupancy_mean_excess_knn_radius': float(
                                    state_occupancy_diagnostics[
                                        'candidate_mean_excess_knn_radius'
                                    ][candidate_idx]
                                ),
                                'occupancy_mean_ratio_to_behavior': float(
                                    state_occupancy_diagnostics[
                                        'candidate_mean_ratio_to_behavior'
                                    ][candidate_idx]
                                ),
                                'online_rank': int(
                                    occupancy_candidate_details[
                                        'online_rank'
                                    ][candidate_idx]
                                ),
                                'fqe_rank': int(
                                    occupancy_candidate_details[
                                        'fqe_rank'
                                    ][candidate_idx]
                                ),
                                'abs_fqe_rank_error': float(
                                    occupancy_candidate_details[
                                        'abs_fqe_rank_error'
                                    ][candidate_idx]
                                ),
                                'abs_fqe_z_error': float(
                                    occupancy_candidate_details[
                                        'abs_fqe_z_error'
                                    ][candidate_idx]
                                ),
                                'occupancy_rank': int(
                                    occupancy_candidate_details[
                                        'occupancy_rank'
                                    ][candidate_idx]
                                ),
                            })

                        print("---------------------------------")
                        print("STATE-OCCUPANCY / FQE ERROR DIAGNOSTIC")
                        print(
                            "occupancy-vs-|FQE rank error| Spearman="
                            f"{occupancy_metrics['occupancy_vs_abs_fqe_rank_error_spearman']:+.4f} | "
                            "OOD-vs-|FQE rank error| Spearman="
                            f"{occupancy_metrics['occupancy_ood_vs_abs_fqe_rank_error_spearman']:+.4f}"
                        )
                        print(
                            "occupancy-vs-|z(FQE)-z(online)| Spearman="
                            f"{occupancy_metrics['occupancy_vs_abs_fqe_z_error_spearman']:+.4f}"
                        )
                        print(
                            "low-novelty quartile mean |rank error|="
                            f"{occupancy_metrics['low_novelty_quartile_mean_abs_rank_error']:.3f} | "
                            "high-novelty quartile="
                            f"{occupancy_metrics['high_novelty_quartile_mean_abs_rank_error']:.3f} | "
                            "difference="
                            f"{occupancy_metrics['high_minus_low_novelty_rank_error']:+.3f}"
                        )
                        print(
                            f"oracle occupancy rank="
                            f"{occupancy_metrics['oracle_occupancy_rank']}/"
                            f"{len(occupancy_raw_fqe)}, "
                            f"oracle OOD fraction="
                            f"{occupancy_metrics['oracle_ood_fraction']:.4f}"
                        )

                        # --------------------------------------------------
                        # TIME-RESOLVED STATE-OCCUPANCY DIAGNOSTIC
                        # --------------------------------------------------
                        if TIME_RESOLVED_OCCUPANCY_STUDY:
                            (
                                time_resolved_rows,
                                time_resolved_candidate_rows,
                            ) = compute_time_resolved_state_occupancy_diagnostics(
                                candidate_trajectory_episodes=(
                                    candidate_occupancy_trajectories
                                ),
                                replay_data=support_reference_data,
                                online_scores=np.asarray(
                                    cum_rews, dtype=np.float64
                                ),
                                raw_fqe_scores=occupancy_raw_fqe,
                                iteration=i,
                                score_metadata=occupancy_fqe_scores,
                            )
                            timeResolvedOccupancyMetrics.extend(
                                time_resolved_rows
                            )
                            timeResolvedOccupancyCandidateRows.extend(
                                time_resolved_candidate_rows
                            )

                            print("---------------------------------")
                            print("TIME-RESOLVED STATE-OCCUPANCY DIAGNOSTIC")
                            for window_row in time_resolved_rows:
                                print(
                                    f"  [{window_row['window_start']},"
                                    f"{window_row['window_end']}): "
                                    f"mean_OOD="
                                    f"{window_row['mean_candidate_ood_fraction']:.4f}, "
                                    f"radius/behavior="
                                    f"{window_row['mean_candidate_radius_ratio_to_behavior']:.3f}, "
                                    f"rho(novelty,online)="
                                    f"{window_row['occupancy_vs_online_spearman']:+.4f}, "
                                    f"rho(novelty,FQE-overvaluation)="
                                    f"{window_row['occupancy_vs_fqe_rank_overvaluation_spearman']:+.4f}"
                                )

                else:
                    replay_coverage_scores = None
                    base_fqe_scores = (
                        sequential_d3rlpy_fqe_scores_preserving_rng(
                            model,
                            agents,
                            fqe_dataset,
                            DIR,
                            i,
                        )
                    )
                    advantage_rew = build_support_penalized_scores(
                        base_fqe_scores=base_fqe_scores,
                        action_divergence=support_action_divergence,
                        support_reference_label=support_reference_label,
                        support_reference_transitions=support_reference_transitions,
                    )

                if len(advantage_rew) != len(agents):
                    raise RuntimeError(
                        "FQE score count mismatch: "
                        f"{len(advantage_rew)} scores for {len(agents)} agents."
                    )

                for j, init_est in enumerate(advantage_rew):
                    if isinstance(advantage_rew, SupportPenalizedFQEScores):
                        print(
                            f"agent{j}: online_return={float(cum_rews[j]):.6f}, "
                            f"FQE={float(advantage_rew.mean_q[j]):.6f}, "
                            f"action_div={float(advantage_rew.action_divergence[j]):.6f}, "
                            f"support_penalty={float(advantage_rew.support_penalty[j]):.6f}, "
                            f"SupportPen_FQE={float(init_est):.6f}"
                        )
                    else:
                        print(
                            f"agent{j}: online_return={float(cum_rews[j]):.6f}, "
                            f"FQE={float(init_est):.6f}"
                        )

            # Candidate evaluation/FQE must not alter the PPO replay buffer. This check guards
            # against accidentally giving FQE access to the online ground-truth trajectories.
            if rank_correlation_study:
                replay_buffer_after_candidate_eval = replay_buffer_signature(model.replay_buffer)
                if replay_buffer_after_candidate_eval != replay_buffer_before_candidate_eval:
                    raise RuntimeError(
                        "Replay buffer changed during candidate evaluation/FQE. This would leak "
                        "online candidate interactions into the offline rank-correlation study."
                    )
                print("Replay-buffer leakage check: PASSED")

            # -----------------------------------------------------------------------------------

            if not online_eval:
                # print(f'ave q losses: {np.mean(q_losses)}, std: {np.std(q_losses)}')
                print(f'ave advantage rew: {np.mean(advantage_rew)}, std: {np.std(advantage_rew)}')
            
            print(f'avg cum rews: {np.mean(cum_rews)}, std: {np.std(cum_rews)}')
            if OBJECTIVE_MISMATCH_STUDY and rank_correlation_study:
                print(
                    f'avg gamma-discounted cum rews: '
                    f'{np.mean(cum_discounted_rews)}, '
                    f'std: {np.std(cum_discounted_rews)}, '
                    f'gamma: {float(model.gamma):.8f}'
                )
            if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                print(f'avg success rate: {np.mean(cum_success):.2f}, std: {np.std(cum_success):.2f}')

            os.makedirs(f'logs/{DIR}', exist_ok=True)

            np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents_to_cpu(agents))
            if online_eval:
                np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)

                if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                    np.save(f'logs/{DIR}/success_{i}_{i + SEARCH_INTERV}.npy', cum_success)
            if not online_eval:
                np.save(f'logs/{DIR}/adv_results_{i}_{i + SEARCH_INTERV}.npy', advantage_rew)
            timeArray.append(time.time() - start_time)

            # Rank-correlation study: compare FQE ranking against the online oracle without
            # affecting the original online selection below. Metrics are computed per iteration
            # because the absolute FQE scale may drift as the offline dataset changes.
            if rank_correlation_study:
                online_scores = np.asarray(cum_rews, dtype=np.float64)
                fqe_mean_q_scores = np.asarray(
                    getattr(advantage_rew, "mean_q", advantage_rew),
                    dtype=np.float64,
                )
                fqe_sigma_scores = np.asarray(
                    getattr(
                        advantage_rew,
                        "mean_sigma",
                        np.zeros(len(advantage_rew), dtype=np.float64),
                    ),
                    dtype=np.float64,
                )
                action_divergence_scores = np.asarray(
                    getattr(
                        advantage_rew,
                        "action_divergence",
                        np.zeros(len(advantage_rew), dtype=np.float64),
                    ),
                    dtype=np.float64,
                )
                support_penalty_scores = np.asarray(
                    getattr(
                        advantage_rew,
                        "support_penalty",
                        np.zeros(len(advantage_rew), dtype=np.float64),
                    ),
                    dtype=np.float64,
                )
                fqe_scores = np.asarray(advantage_rew, dtype=np.float64)

                if len(online_scores) != len(fqe_scores):
                    raise RuntimeError(
                        f"Rank-study length mismatch: {len(online_scores)} online returns vs "
                        f"{len(fqe_scores)} FQE scores."
                    )

                rank_df = pd.DataFrame({
                    'fqe': fqe_scores,
                    'online': online_scores,
                })
                pearson = float(rank_df.corr(method='pearson').loc['fqe', 'online'])
                spearman = float(rank_df.corr(method='spearman').loc['fqe', 'online'])
                kendall = float(rank_df.corr(method='kendall').loc['fqe', 'online'])

                oracle_idx = int(np.argmax(online_scores))
                fqe_idx = int(np.argmax(fqe_scores))
                oracle_return = float(online_scores[oracle_idx])
                fqe_selected_true_return = float(online_scores[fqe_idx])
                selection_regret = float(oracle_return - fqe_selected_true_return)
                online_order = np.argsort(online_scores)[::-1]
                top1_agreement = bool(fqe_idx == oracle_idx)
                top3_hit = bool(fqe_idx in online_order[:min(3, len(online_order))])
                top5_hit = bool(fqe_idx in online_order[:min(5, len(online_order))])

                is_support_penalized = isinstance(
                    advantage_rew, SupportPenalizedFQEScores
                )
                score_label = (
                    "Support-Penalized FQE"
                    if is_support_penalized
                    else "FQE"
                )

                rank_metrics = {
                    'iteration': int(i),
                    'fqe_score_type': (
                        'support_penalized'
                        if is_support_penalized
                        else 'mean'
                    ),
                    'fqe_ensemble_size': int(
                        getattr(advantage_rew, 'ensemble_size', 1)
                    ),
                    'support_penalty_lambda': float(
                        getattr(advantage_rew, 'penalty_lambda', 0.0)
                    ),
                    'support_reference': str(
                        getattr(
                            advantage_rew,
                            'support_reference_label',
                            'none',
                        )
                    ),
                    'support_reference_transitions': int(
                        getattr(
                            advantage_rew,
                            'support_reference_transitions',
                            0,
                        )
                    ),
                    'fqe_objective': str(
                        getattr(advantage_rew, 'fqe_objective', 'unknown')
                    ),
                    'fqe_gamma': float(
                        getattr(advantage_rew, 'fqe_gamma', np.nan)
                    ),
                    'finite_horizon_steps': (
                        -1
                        if getattr(
                            advantage_rew,
                            'finite_horizon_steps',
                            None,
                        ) is None
                        else int(
                            getattr(
                                advantage_rew,
                                'finite_horizon_steps',
                            )
                        )
                    ),
                    'time_conditioned': bool(
                        getattr(advantage_rew, 'time_conditioned', False)
                    ),
                    'fqe_n_steps': int(
                        getattr(advantage_rew, 'fqe_n_steps', FQE_N_STEPS)
                    ),
                    'fqe_target_update_interval': int(
                        getattr(
                            advantage_rew,
                            'fqe_target_update_interval',
                            NATIVE_FQE_TARGET_UPDATE_INTERVAL,
                        )
                    ),
                    'fqe_target_updates': int(
                        getattr(
                            advantage_rew,
                            'fqe_target_updates',
                            ((FQE_N_STEPS - 1) // NATIVE_FQE_TARGET_UPDATE_INTERVAL) + 1,
                        )
                    ),
                    'pearson': pearson,
                    'spearman': spearman,
                    'kendall': kendall,
                    'oracle_idx': oracle_idx,
                    'fqe_idx': fqe_idx,
                    'oracle_return': oracle_return,
                    'fqe_selected_true_return': fqe_selected_true_return,
                    'selection_regret': selection_regret,
                    'top1_agreement': top1_agreement,
                    'top3_hit': top3_hit,
                    'top5_hit': top5_hit,
                }
                rankStudyMetrics.append(rank_metrics)
                for candidate_idx, (fqe_score, online_score) in enumerate(
                    zip(fqe_scores, online_scores)
                ):
                    rankStudyCandidateRows.append({
                        'iteration': int(i),
                        'candidate': int(candidate_idx),
                        # Backward-compatible 'fqe' column is the actual ranking
                        # score: FQE - lambda * behavioral action divergence.
                        'fqe': float(fqe_score),
                        'fqe_mean_q': float(fqe_mean_q_scores[candidate_idx]),
                        'action_divergence': float(
                            action_divergence_scores[candidate_idx]
                        ),
                        'support_penalty': float(
                            support_penalty_scores[candidate_idx]
                        ),
                        'fqe_mean_sigma': float(fqe_sigma_scores[candidate_idx]),
                        'fqe_objective': str(
                            getattr(advantage_rew, 'fqe_objective', 'unknown')
                        ),
                        'fqe_gamma': float(
                            getattr(advantage_rew, 'fqe_gamma', np.nan)
                        ),
                        'finite_horizon_steps': (
                            -1
                            if getattr(
                                advantage_rew,
                                'finite_horizon_steps',
                                None,
                            ) is None
                            else int(
                                getattr(
                                    advantage_rew,
                                    'finite_horizon_steps',
                                )
                            )
                        ),
                        'time_conditioned': bool(
                            getattr(
                                advantage_rew,
                                'time_conditioned',
                                False,
                            )
                        ),
                        'fqe_n_steps': int(
                            getattr(advantage_rew, 'fqe_n_steps', FQE_N_STEPS)
                        ),
                        'fqe_target_update_interval': int(
                            getattr(
                                advantage_rew,
                                'fqe_target_update_interval',
                                NATIVE_FQE_TARGET_UPDATE_INTERVAL,
                            )
                        ),
                        'fqe_target_updates': int(
                            getattr(
                                advantage_rew,
                                'fqe_target_updates',
                                ((FQE_N_STEPS - 1) // NATIVE_FQE_TARGET_UPDATE_INTERVAL) + 1,
                            )
                        ),
                        'online': float(online_score),
                    })

                print("---------------------------------")
                print(f"{score_label} / ONLINE RANKING STUDY")
                print(f"Pearson correlation:  {pearson:.4f}")
                print(f"Spearman correlation: {spearman:.4f}")
                print(f"Kendall tau:          {kendall:.4f}")
                print(f"Online best agent:    {oracle_idx}")
                print(f"{score_label} best agent:   {fqe_idx}")
                print(f"Online best return:   {oracle_return:.4f}")
                print(
                    f"{score_label}-selected agent true return: "
                    f"{fqe_selected_true_return:.4f}"
                )
                print(f"Selection regret:     {selection_regret:.4f}")
                print(f"Exact top-1 agreement:       {top1_agreement}")
                print(f"{score_label} choice in online top-3:  {top3_hit}")
                print(f"{score_label} choice in online top-5:  {top5_hit}")

                # Save raw paired scores and per-iteration metrics for later analysis.
                np.save(
                    f'logs/{DIR}/fqe_results_{i}_{i + SEARCH_INTERV}.npy',
                    fqe_scores
                )
                # Save the raw components so lambda can be swept post-hoc
                # without rerunning FQE or online candidate evaluation. Existing
                # fqe_results_* remains the actual ranking score for backward
                # compatibility.
                np.save(
                    f'logs/{DIR}/fqe_mean_q_results_{i}_{i + SEARCH_INTERV}.npy',
                    fqe_mean_q_scores
                )
                np.save(
                    f'logs/{DIR}/fqe_action_divergence_results_{i}_{i + SEARCH_INTERV}.npy',
                    action_divergence_scores
                )
                np.save(
                    f'logs/{DIR}/fqe_support_penalty_results_{i}_{i + SEARCH_INTERV}.npy',
                    support_penalty_scores
                )
                np.save(
                    f'logs/{DIR}/fqe_sigma_results_{i}_{i + SEARCH_INTERV}.npy',
                    fqe_sigma_scores
                )
                if hasattr(advantage_rew, 'ensemble_member_values'):
                    np.save(
                        f'logs/{DIR}/fqe_ensemble_member_values_{i}_{i + SEARCH_INTERV}.npy',
                        advantage_rew.ensemble_member_values
                    )
                np.save(
                    f'logs/{DIR}/online_all_results_{i}_{i + SEARCH_INTERV}.npy',
                    online_scores
                )
                if OBJECTIVE_MISMATCH_STUDY:
                    np.save(
                        f'logs/{DIR}/online_discounted_all_results_'
                        f'{i}_{i + SEARCH_INTERV}.npy',
                        np.asarray(cum_discounted_rews, dtype=np.float64),
                    )
                np.save(
                    f'logs/{DIR}/rank_metrics_{i}_{i + SEARCH_INTERV}.npy',
                    rank_metrics
                )

                if KNN_SUPPORT_STUDY and knn_support_diagnostics is not None:
                    np.save(
                        f'logs/{DIR}/knn_support_unsupported_fraction_'
                        f'{i}_{i + SEARCH_INTERV}.npy',
                        knn_support_diagnostics[
                            'candidate_unsupported_fraction'
                        ],
                    )
                    np.save(
                        f'logs/{DIR}/knn_support_mean_sq_l2_'
                        f'{i}_{i + SEARCH_INTERV}.npy',
                        knn_support_diagnostics['candidate_mean_sq_l2'],
                    )

                # --------------------------------------------------------------
                # OBJECTIVE MISMATCH STUDY (analysis only)
                # --------------------------------------------------------------
                # Compare RAW ordinary FQE (ensemble mean Q) to two online
                # targets measured on the exact same evaluation trajectories:
                #   1) original undiscounted episodic return (canonical selector)
                #   2) PPO-gamma-discounted episodic return (historical diagnostic)
                #
                # With time-conditioned finite-horizon FQE enabled, target (1)
                # is now the OBJECTIVE-ALIGNED comparison; target (2) remains
                # only to show how the new evaluator differs from the previous
                # discounted FQE objective. The support-penalized score is
                # intentionally NOT used here.
                if OBJECTIVE_MISMATCH_STUDY:
                    discounted_online_scores = np.asarray(
                        cum_discounted_rews, dtype=np.float64
                    )
                    if len(discounted_online_scores) != len(online_scores):
                        raise RuntimeError(
                            "Objective-mismatch diagnostic did not collect one "
                            "discounted return per candidate: "
                            f"discounted={len(discounted_online_scores)}, "
                            f"undiscounted={len(online_scores)}."
                        )

                    objective_metrics = compute_objective_mismatch_metrics(
                        online_undiscounted=online_scores,
                        online_discounted=discounted_online_scores,
                        fqe_mean_q=fqe_mean_q_scores,
                        iteration=i,
                        gamma=model.gamma,
                    )
                    objective_metrics["fqe_objective"] = str(
                        getattr(advantage_rew, "fqe_objective", "unknown")
                    )
                    objective_metrics["fqe_gamma"] = float(
                        getattr(advantage_rew, "fqe_gamma", np.nan)
                    )
                    objective_metrics["finite_horizon_steps"] = (
                        -1
                        if getattr(
                            advantage_rew,
                            "finite_horizon_steps",
                            None,
                        ) is None
                        else int(
                            getattr(
                                advantage_rew,
                                "finite_horizon_steps",
                            )
                        )
                    )
                    objective_metrics["time_conditioned"] = bool(
                        getattr(advantage_rew, "time_conditioned", False)
                    )
                    objectiveMismatchMetrics.append(objective_metrics)

                    for candidate_idx in range(len(online_scores)):
                        objectiveMismatchCandidateRows.append({
                            "iteration": int(i),
                            "candidate": int(candidate_idx),
                            "gamma": float(model.gamma),
                            "fqe_objective": str(
                                getattr(
                                    advantage_rew,
                                    "fqe_objective",
                                    "unknown",
                                )
                            ),
                            "fqe_gamma": float(
                                getattr(
                                    advantage_rew,
                                    "fqe_gamma",
                                    np.nan,
                                )
                            ),
                            "finite_horizon_steps": (
                                -1
                                if getattr(
                                    advantage_rew,
                                    "finite_horizon_steps",
                                    None,
                                ) is None
                                else int(
                                    getattr(
                                        advantage_rew,
                                        "finite_horizon_steps",
                                    )
                                )
                            ),
                            "time_conditioned": bool(
                                getattr(
                                    advantage_rew,
                                    "time_conditioned",
                                    False,
                                )
                            ),
                            "fqe_mean_q": float(
                                fqe_mean_q_scores[candidate_idx]
                            ),
                            "online_undiscounted": float(
                                online_scores[candidate_idx]
                            ),
                            "online_discounted": float(
                                discounted_online_scores[candidate_idx]
                            ),
                        })

                    print("---------------------------------")
                    print("FQE OBJECTIVE-ALIGNMENT DIAGNOSTIC")
                    print(
                        "FQE evaluator: "
                        f"{objective_metrics['fqe_objective']} | "
                        f"OPE gamma={objective_metrics['fqe_gamma']:.8f} | "
                        f"H={objective_metrics['finite_horizon_steps']} | "
                        f"time_conditioned="
                        f"{objective_metrics['time_conditioned']}"
                    )
                    print(
                        "PPO / discounted-online diagnostic gamma: "
                        f"{float(model.gamma):.8f}"
                    )
                    print(
                        "FQE vs UNDISCOUNTED online: "
                        f"Pearson="
                        f"{objective_metrics['fqe_vs_undiscounted_pearson']:+.4f}, "
                        f"Spearman="
                        f"{objective_metrics['fqe_vs_undiscounted_spearman']:+.4f}, "
                        f"Kendall="
                        f"{objective_metrics['fqe_vs_undiscounted_kendall']:+.4f}"
                    )
                    print(
                        "FQE vs DISCOUNTED online:   "
                        f"Pearson="
                        f"{objective_metrics['fqe_vs_discounted_pearson']:+.4f}, "
                        f"Spearman="
                        f"{objective_metrics['fqe_vs_discounted_spearman']:+.4f}, "
                        f"Kendall="
                        f"{objective_metrics['fqe_vs_discounted_kendall']:+.4f}"
                    )
                    print(
                        "Discounted - undiscounted correlation delta: "
                        f"Pearson="
                        f"{objective_metrics['discounted_minus_undiscounted_pearson']:+.4f}, "
                        f"Spearman="
                        f"{objective_metrics['discounted_minus_undiscounted_spearman']:+.4f}, "
                        f"Kendall="
                        f"{objective_metrics['discounted_minus_undiscounted_kendall']:+.4f}"
                    )
                    print(
                        "Online objective agreement: "
                        f"Spearman="
                        f"{objective_metrics['online_objectives_spearman']:+.4f}, "
                        f"same top-1="
                        f"{objective_metrics['online_oracle_top1_same']}"
                    )
                    print(
                        "Raw FQE top-1 agreement: "
                        f"undiscounted="
                        f"{objective_metrics['fqe_top1_undiscounted']}, "
                        f"discounted="
                        f"{objective_metrics['fqe_top1_discounted']}"
                    )
                    print(
                        "Raw FQE selection regret: "
                        f"undiscounted="
                        f"{objective_metrics['fqe_regret_undiscounted']:.4f}, "
                        f"discounted="
                        f"{objective_metrics['fqe_regret_discounted']:.4f}"
                    )

                    np.save(
                        f'logs/{DIR}/objective_mismatch_metrics_'
                        f'{i}_{i + SEARCH_INTERV}.npy',
                        objective_metrics,
                    )

            # Replay-coverage ablation metrics are analysis-only. The canonical
            # rankStudyMetrics above remain the FULL-buffer scores exactly as before.
            if (
                rank_correlation_study
                and FQE_BACKEND == "native_batched"
                and REPLAY_COVERAGE_ABLATION
            ):
                online_scores_for_coverage = np.asarray(
                    cum_rews, dtype=np.float64
                )

                print("---------------------------------")
                print("REPLAY COVERAGE ABLATION RESULTS")

                for coverage_label, coverage_scores in replay_coverage_scores.items():
                    coverage_data = replay_coverage_data[coverage_label]
                    coverage_metrics = compute_replay_coverage_rank_metrics(
                        online_scores=online_scores_for_coverage,
                        fqe_scores=coverage_scores,
                        iteration=i,
                        coverage_label=coverage_label,
                        native_data=coverage_data,
                    )
                    replayCoverageMetrics.append(coverage_metrics)

                    coverage_scores_np = np.asarray(
                        coverage_scores, dtype=np.float64
                    )
                    coverage_mean_q_np = np.asarray(
                        getattr(coverage_scores, "mean_q", coverage_scores_np),
                        dtype=np.float64,
                    )
                    coverage_sigma_np = np.asarray(
                        getattr(
                            coverage_scores,
                            "mean_sigma",
                            np.zeros_like(coverage_scores_np),
                        ),
                        dtype=np.float64,
                    )
                    coverage_action_div_np = np.asarray(
                        getattr(
                            coverage_scores,
                            "action_divergence",
                            np.zeros_like(coverage_scores_np),
                        ),
                        dtype=np.float64,
                    )
                    coverage_support_penalty_np = np.asarray(
                        getattr(
                            coverage_scores,
                            "support_penalty",
                            np.zeros_like(coverage_scores_np),
                        ),
                        dtype=np.float64,
                    )
                    for candidate_idx, (coverage_fqe_score, online_score) in enumerate(
                        zip(coverage_scores_np, online_scores_for_coverage)
                    ):
                        replayCoverageCandidateRows.append({
                            "iteration": int(i),
                            "coverage": str(coverage_label),
                            "requested_max_transitions": (
                                -1
                                if coverage_data["requested_max_transitions"] is None
                                else int(coverage_data["requested_max_transitions"])
                            ),
                            "actual_transitions": int(coverage_data["actual_transitions"]),
                            "n_episodes": int(coverage_data["n_episodes"]),
                            "training_initial_states": int(
                                coverage_data.get(
                                    "training_initial_states",
                                    len(coverage_data["initial_observations"]),
                                )
                            ),
                            "score_initial_states": int(
                                coverage_data.get(
                                    "score_initial_states",
                                    len(coverage_data["initial_observations"]),
                                )
                            ),
                            "candidate": int(candidate_idx),
                            # 'fqe' remains the actual support-penalized
                            # ranking score for backward compatibility.
                            "fqe": float(coverage_fqe_score),
                            "fqe_mean_q": float(
                                coverage_mean_q_np[candidate_idx]
                            ),
                            "action_divergence": float(
                                coverage_action_div_np[candidate_idx]
                            ),
                            "support_penalty": float(
                                coverage_support_penalty_np[candidate_idx]
                            ),
                            "fqe_mean_sigma": float(
                                coverage_sigma_np[candidate_idx]
                            ),
                            "fqe_ensemble_size": int(
                                getattr(coverage_scores, "ensemble_size", 1)
                            ),
                            "support_penalty_lambda": float(
                                getattr(
                                    coverage_scores,
                                    "penalty_lambda",
                                    0.0,
                                )
                            ),
                            "support_reference": str(
                                getattr(
                                    coverage_scores,
                                    "support_reference_label",
                                    "none",
                                )
                            ),
                            "support_reference_transitions": int(
                                getattr(
                                    coverage_scores,
                                    "support_reference_transitions",
                                    0,
                                )
                            ),
                            "fqe_objective": str(
                                getattr(
                                    coverage_scores,
                                    "fqe_objective",
                                    "unknown",
                                )
                            ),
                            "fqe_gamma": float(
                                getattr(
                                    coverage_scores,
                                    "fqe_gamma",
                                    np.nan,
                                )
                            ),
                            "finite_horizon_steps": (
                                -1
                                if getattr(
                                    coverage_scores,
                                    "finite_horizon_steps",
                                    None,
                                ) is None
                                else int(
                                    getattr(
                                        coverage_scores,
                                        "finite_horizon_steps",
                                    )
                                )
                            ),
                            "time_conditioned": bool(
                                getattr(
                                    coverage_scores,
                                    "time_conditioned",
                                    False,
                                )
                            ),
                            "online": float(online_score),
                        })

                    direct_summary = (
                        f"coverage={coverage_label:>5} | "
                        f"actual={coverage_metrics['actual_transitions']:>6} | "
                        f"episodes={coverage_metrics['n_episodes']:>3} | "
                        f"score_s0={coverage_metrics['score_initial_states']:>3} | "
                        f"Pearson={coverage_metrics['pearson']:+.4f} | "
                        f"Spearman={coverage_metrics['spearman']:+.4f} | "
                        f"Kendall={coverage_metrics['kendall']:+.4f} | "
                        f"direct_top1={int(coverage_metrics['top1_agreement'])} | "
                        f"direct_regret={coverage_metrics['selection_regret']:.4f}"
                    )

                    hybrid_parts = []
                    for requested_k in HYBRID_TOPK_VALUES:
                        hybrid_parts.append(
                            f"Recall@{requested_k}="
                            f"{int(coverage_metrics[f'oracle_recall_at_{requested_k}'])}, "
                            f"HReg@{requested_k}="
                            f"{coverage_metrics[f'hybrid_regret_at_{requested_k}']:.4f}"
                        )

                    print(
                        direct_summary
                        + " | "
                        + " | ".join(hybrid_parts)
                    )

                replay_coverage_npz_payload = {
                    "online": online_scores_for_coverage,
                }
                for coverage_label, scores in replay_coverage_scores.items():
                    # Backward-compatible key: fqe_<window> is the actual
                    # support-penalized ranking score. Save both raw components
                    # so lambda can be swept post-hoc.
                    replay_coverage_npz_payload[
                        f"fqe_{coverage_label}"
                    ] = np.asarray(scores, dtype=np.float64)
                    replay_coverage_npz_payload[
                        f"fqe_mean_q_{coverage_label}"
                    ] = np.asarray(
                        getattr(scores, "mean_q", scores),
                        dtype=np.float64,
                    )
                    replay_coverage_npz_payload[
                        f"action_divergence_{coverage_label}"
                    ] = np.asarray(
                        getattr(
                            scores,
                            "action_divergence",
                            np.zeros(len(scores), dtype=np.float64),
                        ),
                        dtype=np.float64,
                    )
                    replay_coverage_npz_payload[
                        f"support_penalty_{coverage_label}"
                    ] = np.asarray(
                        getattr(
                            scores,
                            "support_penalty",
                            np.zeros(len(scores), dtype=np.float64),
                        ),
                        dtype=np.float64,
                    )
                    replay_coverage_npz_payload[
                        f"fqe_sigma_{coverage_label}"
                    ] = np.asarray(
                        getattr(
                            scores,
                            "mean_sigma",
                            np.zeros(len(scores), dtype=np.float64),
                        ),
                        dtype=np.float64,
                    )

                np.savez(
                    f'logs/{DIR}/replay_coverage_scores_{i}_{i + SEARCH_INTERV}.npz',
                    **replay_coverage_npz_payload,
                )

            # Correlation calculation used by the original offline-selection path.
            if not online_eval:
                df = pd.DataFrame({
                    'advantage': advantage_rew,
                    'online': cum_rews
                })
                corr_pear = df.corr(method='pearson')
                corr_spearman = df.corr(method='spearman')
                corr_kendall = df.corr(method='kendall')
                print("Pearson correlation coefficient:", corr_pear['advantage'][1])
                print("Spearman correlation coefficient:", corr_spearman['advantage'][1])
                print("Kendall Tau correlation coefficient:", corr_kendall['advantage']['online'])

                # Code using Advantage estimiation

                # Using the best agent from the top 5
                # top_5_idx = np.argsort(advantage_rew)[-5:]
                # top_5_agents = np.array(agents)[top_5_idx]
                # best_agent, best_idx, returns_trains = None, None, -float('inf')
                # dummy_env = gym.make(env_name)
                # for idx, tagent in enumerate(top_5_agents):
                #     model.policy.load_state_dict(tagent)
                #     model.policy.to(device)
                #     cur_return = evaluate_policy(model, dummy_env, n_eval_episodes=2, callback=evaluation_callback, deterministic=True)[0]
                #     if cur_return > returns_trains:
                #         returns_trains = cur_return
                #         best_idx = idx
                # best_agent = top_5_agents[best_idx]

                # print(f'the best agent: {best_idx}, avg policy: {returns_trains}')
                # best_agent_index.append(best_idx)
                # np.save(f'logs/{DIR}/best_agent_{i}_{i + SEARCH_INTERV}.npy', best_agent_index)
                # load_state_dict(model, best_agent)

                # -----------------------------------------------------------------------------

                # Code using d3rlpy FQE

                best_idx = np.argsort(advantage_rew)[-1]
                best_agent = agents[best_idx]
                print(f'the best agent: {best_idx}, best agent cum rewards: {cum_rews[best_idx]}')
                best_agent_index.append(best_idx)
                np.save(f'logs/{DIR}/best_agent_{i}_{i + SEARCH_INTERV}.npy', best_agent_index)
                np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews[best_idx])
                load_state_dict(model, best_agent)

            # Finding the best agent from online evaluation
            if online_eval:
                if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                    # Mask for successes. Keep the original selection rule, but convert the
                    # Python lists to arrays before boolean indexing.
                    success_array = np.asarray(cum_success)
                    reward_array = np.asarray(cum_rews)
                    success_mask = success_array == 1.0

                    if np.any(success_mask):  # only true if there's at least one success
                        successful_rews = reward_array[success_mask]
                        best_idx_in_success = np.argmax(successful_rews)  # first occurrence of max
                        best_idx = np.where(success_mask)[0][best_idx_in_success]
                    else:
                        # No successes at all → fallback to best reward overall
                        best_idx = np.argmax(cum_rews)
                else:
                    best_idx = np.argsort(cum_rews)[-1]

                best_agent = agents[best_idx]
                print(f'the best agent: {best_idx}, best agent cum rewards: {cum_rews[best_idx]}, best agent success rate: {cum_success[best_idx] if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"] else "N/A"}')
                best_agent_index.append(best_idx)
                np.save(f'logs/{DIR}/best_agent_{i}_{i + SEARCH_INTERV}.npy', best_agent_index)
                load_state_dict(model, best_agent)

        if rank_correlation_study and rankStudyMetrics:
            rank_summary_df = pd.DataFrame(rankStudyMetrics)
            rank_summary_df.to_csv(f'logs/{DIR}/rank_study_summary.csv', index=False)
            pd.DataFrame(rankStudyCandidateRows).to_csv(
                f'logs/{DIR}/rank_study_candidates.csv', index=False
            )
            np.save(
                f'logs/{DIR}/rank_study_summary.npy',
                np.array(rankStudyMetrics, dtype=object),
                allow_pickle=True
            )

            print("---------------------------------")
            summary_score_label = (
                "Support-Penalized FQE"
                if (
                    rank_summary_df["fqe_score_type"]
                    == "support_penalized"
                ).all()
                else "FQE"
            )
            print(f"{summary_score_label} / ONLINE RANKING STUDY SUMMARY")
            print(
                f"Mean Pearson:  {rank_summary_df['pearson'].mean():.4f} "
                f"+/- {rank_summary_df['pearson'].std(ddof=0):.4f}"
            )
            print(
                f"Mean Spearman: {rank_summary_df['spearman'].mean():.4f} "
                f"+/- {rank_summary_df['spearman'].std(ddof=0):.4f}"
            )
            print(
                f"Mean Kendall:  {rank_summary_df['kendall'].mean():.4f} "
                f"+/- {rank_summary_df['kendall'].std(ddof=0):.4f}"
            )
            print(
                f"Top-1 agreement rate: "
                f"{rank_summary_df['top1_agreement'].mean():.3f}"
            )
            print(
                f"Top-3 hit rate: "
                f"{rank_summary_df['top3_hit'].mean():.3f}"
            )
            print(
                f"Top-5 hit rate: "
                f"{rank_summary_df['top5_hit'].mean():.3f}"
            )
            print(
                f"Mean selection regret: "
                f"{rank_summary_df['selection_regret'].mean():.4f}"
            )

        if OBJECTIVE_MISMATCH_STUDY and objectiveMismatchMetrics:
            objective_summary_df = pd.DataFrame(objectiveMismatchMetrics)
            objective_candidates_df = pd.DataFrame(
                objectiveMismatchCandidateRows
            )

            objective_summary_df.to_csv(
                f'logs/{DIR}/objective_mismatch_summary.csv',
                index=False,
            )
            objective_candidates_df.to_csv(
                f'logs/{DIR}/objective_mismatch_candidates.csv',
                index=False,
            )
            np.save(
                f'logs/{DIR}/objective_mismatch_summary.npy',
                np.array(objectiveMismatchMetrics, dtype=object),
                allow_pickle=True,
            )

            print("---------------------------------")
            print("FQE OBJECTIVE-ALIGNMENT STUDY SUMMARY")
            if "fqe_objective" in objective_summary_df.columns:
                print(
                    "FQE evaluator: "
                    f"{objective_summary_df['fqe_objective'].iloc[0]} | "
                    f"OPE gamma="
                    f"{objective_summary_df['fqe_gamma'].iloc[0]:.8f} | "
                    f"H="
                    f"{int(objective_summary_df['finite_horizon_steps'].iloc[0])} | "
                    f"time_conditioned="
                    f"{bool(objective_summary_df['time_conditioned'].iloc[0])}"
                )
            print(
                "PPO / discounted-online diagnostic gamma: "
                f"{objective_summary_df['gamma'].iloc[0]:.8f}"
            )
            print(
                "FQE vs UNDISCOUNTED online -- "
                f"Pearson: "
                f"{objective_summary_df['fqe_vs_undiscounted_pearson'].mean():+.4f} "
                f"+/- "
                f"{objective_summary_df['fqe_vs_undiscounted_pearson'].std(ddof=0):.4f} | "
                f"Spearman: "
                f"{objective_summary_df['fqe_vs_undiscounted_spearman'].mean():+.4f} "
                f"+/- "
                f"{objective_summary_df['fqe_vs_undiscounted_spearman'].std(ddof=0):.4f} | "
                f"Kendall: "
                f"{objective_summary_df['fqe_vs_undiscounted_kendall'].mean():+.4f} "
                f"+/- "
                f"{objective_summary_df['fqe_vs_undiscounted_kendall'].std(ddof=0):.4f}"
            )
            print(
                "FQE vs DISCOUNTED online   -- "
                f"Pearson: "
                f"{objective_summary_df['fqe_vs_discounted_pearson'].mean():+.4f} "
                f"+/- "
                f"{objective_summary_df['fqe_vs_discounted_pearson'].std(ddof=0):.4f} | "
                f"Spearman: "
                f"{objective_summary_df['fqe_vs_discounted_spearman'].mean():+.4f} "
                f"+/- "
                f"{objective_summary_df['fqe_vs_discounted_spearman'].std(ddof=0):.4f} | "
                f"Kendall: "
                f"{objective_summary_df['fqe_vs_discounted_kendall'].mean():+.4f} "
                f"+/- "
                f"{objective_summary_df['fqe_vs_discounted_kendall'].std(ddof=0):.4f}"
            )
            print(
                "Mean DISCOUNTED - UNDISCOUNTED correlation difference -- "
                f"Pearson: "
                f"{objective_summary_df['discounted_minus_undiscounted_pearson'].mean():+.4f} | "
                f"Spearman: "
                f"{objective_summary_df['discounted_minus_undiscounted_spearman'].mean():+.4f} | "
                f"Kendall: "
                f"{objective_summary_df['discounted_minus_undiscounted_kendall'].mean():+.4f}"
            )
            print(
                "Raw FQE top-1 agreement -- "
                f"undiscounted: "
                f"{objective_summary_df['fqe_top1_undiscounted'].mean():.3f} | "
                f"discounted: "
                f"{objective_summary_df['fqe_top1_discounted'].mean():.3f}"
            )
            print(
                "Online discounted/undiscounted oracle top-1 same rate: "
                f"{objective_summary_df['online_oracle_top1_same'].mean():.3f}"
            )
            print(
                "Mean raw-FQE selection regret -- "
                f"undiscounted: "
                f"{objective_summary_df['fqe_regret_undiscounted'].mean():.4f} | "
                f"discounted: "
                f"{objective_summary_df['fqe_regret_discounted'].mean():.4f}"
            )

        if FQE_CONVERGENCE_STUDY and fqeConvergenceMetrics:
            convergence_summary_df = pd.DataFrame(fqeConvergenceMetrics)
            convergence_candidates_df = pd.DataFrame(
                fqeConvergenceCandidateRows
            )

            convergence_summary_df.to_csv(
                f'logs/{DIR}/fqe_convergence_summary.csv',
                index=False,
            )
            convergence_candidates_df.to_csv(
                f'logs/{DIR}/fqe_convergence_candidates.csv',
                index=False,
            )
            np.save(
                f'logs/{DIR}/fqe_convergence_summary.npy',
                np.array(fqeConvergenceMetrics, dtype=object),
                allow_pickle=True,
            )

            print("---------------------------------")
            print("FQE LONG-HORIZON CONVERGENCE STUDY SUMMARY")
            for conv_steps, conv_target_interval in FQE_CONVERGENCE_CONFIGS:
                config_label = (
                    f"{int(conv_steps)}_steps_target"
                    f"{int(conv_target_interval)}"
                )
                group = convergence_summary_df[
                    convergence_summary_df["config"] == config_label
                ]
                if group.empty:
                    continue

                print(
                    f"steps={int(conv_steps):>6}, "
                    f"target={int(conv_target_interval):>3}, "
                    f"target_updates={int(group['fqe_target_updates'].iloc[0]):>5} | "
                    f"meanQ={group['mean_fqe_q'].mean():.3f} "
                    f"+/- {group['mean_fqe_q'].std(ddof=0):.3f} | "
                    f"Qrange={group['fqe_q_range'].mean():.3f} | "
                    f"final_loss={group['mean_final_fqe_loss'].mean():.4f} | "
                    f"online_mean={group['mean_online_return'].mean():.3f} | "
                    f"Pearson={group['pearson'].mean():+.4f} "
                    f"+/- {group['pearson'].std(ddof=0):.4f} | "
                    f"Spearman={group['spearman'].mean():+.4f} "
                    f"+/- {group['spearman'].std(ddof=0):.4f} | "
                    f"Kendall={group['kendall'].mean():+.4f} "
                    f"+/- {group['kendall'].std(ddof=0):.4f} | "
                    f"top1={group['top1_agreement'].mean():.3f} | "
                    f"top3={group['top3_hit'].mean():.3f} | "
                    f"top5={group['top5_hit'].mean():.3f} | "
                    f"regret={group['selection_regret'].mean():.4f}"
                )

        if KNN_SUPPORT_STUDY and knnSupportMetrics:
            knn_summary_df = pd.DataFrame(knnSupportMetrics)
            knn_candidates_df = pd.DataFrame(knnSupportCandidateRows)

            knn_summary_df.to_csv(
                f'logs/{DIR}/knn_support_summary.csv',
                index=False,
            )
            knn_candidates_df.to_csv(
                f'logs/{DIR}/knn_support_candidates.csv',
                index=False,
            )
            np.save(
                f'logs/{DIR}/knn_support_summary.npy',
                np.array(knnSupportMetrics, dtype=object),
                allow_pickle=True,
            )

            print("---------------------------------")
            print("STATE-CONDITIONAL kNN SUPPORT FILTER SUMMARY")
            print(
                f"FQE config: "
                f"{int(knn_summary_df['fqe_n_steps'].iloc[0])} steps / "
                f"target "
                f"{int(knn_summary_df['fqe_target_update_interval'].iloc[0])}"
            )
            print(
                f"k={int(knn_summary_df['knn_k'].iloc[0])}, "
                f"mean query states="
                f"{knn_summary_df['query_states'].mean():.1f}, "
                f"behavior percentile="
                f"{knn_summary_df['behavior_percentile'].iloc[0]:.1f}, "
                f"max unsupported fraction="
                f"{knn_summary_df['max_unsupported_fraction'].iloc[0]:.3f}, "
                f"min keep={int(knn_summary_df['min_keep'].iloc[0])}"
            )
            print(
                "Mean raw 50k/100 FQE Spearman: "
                f"{knn_summary_df['raw_fqe_spearman'].mean():+.4f} "
                f"+/- {knn_summary_df['raw_fqe_spearman'].std(ddof=0):.4f}"
            )
            print(
                "Mean support-score Spearman vs online: "
                f"{knn_summary_df['support_score_spearman'].mean():+.4f} "
                f"+/- "
                f"{knn_summary_df['support_score_spearman'].std(ddof=0):.4f}"
            )
            print(
                "Mean absolute-threshold keep count: "
                f"{knn_summary_df['threshold_keep_count'].mean():.2f} / "
                f"{len(agents)}"
            )
            print(
                "Mean effective keep count: "
                f"{knn_summary_df['effective_keep_count'].mean():.2f} / "
                f"{len(agents)} | "
                f"mean fallback additions="
                f"{knn_summary_df['fallback_fill_count'].mean():.2f}"
            )
            print(
                "Oracle survives effective support filter: "
                f"{knn_summary_df['oracle_survives_effective_filter'].mean():.3f}"
            )
            print(
                "Direct raw FQE top-1 / regret: "
                f"{knn_summary_df['raw_fqe_top1'].mean():.3f} / "
                f"{knn_summary_df['raw_fqe_selection_regret'].mean():.4f}"
            )
            print(
                "Direct filtered-FQE top-1 / regret: "
                f"{knn_summary_df['filtered_fqe_top1'].mean():.3f} / "
                f"{knn_summary_df['filtered_fqe_selection_regret'].mean():.4f}"
            )
            for requested_k in HYBRID_TOPK_VALUES:
                print(
                    f"k={requested_k}: raw Recall="
                    f"{knn_summary_df[f'raw_oracle_recall_at_{requested_k}'].mean():.3f}, "
                    f"raw HReg="
                    f"{knn_summary_df[f'raw_hybrid_regret_at_{requested_k}'].mean():.4f} | "
                    f"filtered Recall="
                    f"{knn_summary_df[f'filtered_oracle_recall_at_{requested_k}'].mean():.3f}, "
                    f"filtered HReg="
                    f"{knn_summary_df[f'filtered_hybrid_regret_at_{requested_k}'].mean():.4f}, "
                    f"effective_k="
                    f"{knn_summary_df[f'filtered_effective_k_at_{requested_k}'].mean():.2f}"
                )

        if STATE_OCCUPANCY_KNN_STUDY and stateOccupancyMetrics:
            occupancy_summary_df = pd.DataFrame(stateOccupancyMetrics)
            occupancy_candidates_df = pd.DataFrame(
                stateOccupancyCandidateRows
            )

            occupancy_summary_df.to_csv(
                f'logs/{DIR}/state_occupancy_knn_summary.csv',
                index=False,
            )
            occupancy_candidates_df.to_csv(
                f'logs/{DIR}/state_occupancy_knn_candidates.csv',
                index=False,
            )
            np.save(
                f'logs/{DIR}/state_occupancy_knn_summary.npy',
                np.array(stateOccupancyMetrics, dtype=object),
                allow_pickle=True,
            )

            print("---------------------------------")
            print("STATE-OCCUPANCY kNN DIAGNOSTIC SUMMARY")
            print(
                f"FQE config: "
                f"{int(occupancy_summary_df['fqe_n_steps'].iloc[0])} "
                f"steps / target "
                f"{int(occupancy_summary_df['fqe_target_update_interval'].iloc[0])}"
            )
            print(
                f"k={int(occupancy_summary_df['knn_k'].iloc[0])}, "
                f"time_aware="
                f"{bool(occupancy_summary_df['include_time'].iloc[0])}, "
                f"mean replay queries="
                f"{occupancy_summary_df['replay_query_states'].mean():.1f}, "
                f"behavior percentile="
                f"{occupancy_summary_df['behavior_percentile'].iloc[0]:.1f}"
            )
            print(
                "Mean candidate occupancy OOD fraction: "
                f"{occupancy_summary_df['mean_candidate_ood_fraction'].mean():.4f} "
                f"+/- "
                f"{occupancy_summary_df['mean_candidate_ood_fraction'].std(ddof=0):.4f}"
            )
            print(
                "Occupancy radius vs |FQE rank error| Spearman: "
                f"{occupancy_summary_df['occupancy_vs_abs_fqe_rank_error_spearman'].mean():+.4f} "
                f"+/- "
                f"{occupancy_summary_df['occupancy_vs_abs_fqe_rank_error_spearman'].std(ddof=0):.4f}"
            )
            print(
                "Occupancy OOD fraction vs |FQE rank error| Spearman: "
                f"{occupancy_summary_df['occupancy_ood_vs_abs_fqe_rank_error_spearman'].mean():+.4f} "
                f"+/- "
                f"{occupancy_summary_df['occupancy_ood_vs_abs_fqe_rank_error_spearman'].std(ddof=0):.4f}"
            )
            print(
                "Occupancy radius vs |z(FQE)-z(online)| Spearman: "
                f"{occupancy_summary_df['occupancy_vs_abs_fqe_z_error_spearman'].mean():+.4f} "
                f"+/- "
                f"{occupancy_summary_df['occupancy_vs_abs_fqe_z_error_spearman'].std(ddof=0):.4f}"
            )
            print(
                "Occupancy radius vs online return Spearman: "
                f"{occupancy_summary_df['occupancy_vs_online_spearman'].mean():+.4f} "
                f"+/- "
                f"{occupancy_summary_df['occupancy_vs_online_spearman'].std(ddof=0):.4f}"
            )
            print(
                "Mean |FQE rank error|, low-novelty quartile / "
                "high-novelty quartile: "
                f"{occupancy_summary_df['low_novelty_quartile_mean_abs_rank_error'].mean():.3f} / "
                f"{occupancy_summary_df['high_novelty_quartile_mean_abs_rank_error'].mean():.3f} "
                f"(high-low="
                f"{occupancy_summary_df['high_minus_low_novelty_rank_error'].mean():+.3f})"
            )
            print(
                "Mean oracle occupancy rank: "
                f"{occupancy_summary_df['oracle_occupancy_rank'].mean():.2f} / "
                f"{len(agents)}"
            )

        if (
            TIME_RESOLVED_OCCUPANCY_STUDY
            and timeResolvedOccupancyMetrics
        ):
            time_occ_summary_df = pd.DataFrame(
                timeResolvedOccupancyMetrics
            )
            time_occ_candidates_df = pd.DataFrame(
                timeResolvedOccupancyCandidateRows
            )

            time_occ_summary_df.to_csv(
                f'logs/{DIR}/time_resolved_occupancy_summary.csv',
                index=False,
            )
            time_occ_candidates_df.to_csv(
                f'logs/{DIR}/time_resolved_occupancy_candidates.csv',
                index=False,
            )
            np.save(
                f'logs/{DIR}/time_resolved_occupancy_summary.npy',
                np.array(timeResolvedOccupancyMetrics, dtype=object),
                allow_pickle=True,
            )

            print("---------------------------------")
            print("TIME-RESOLVED STATE-OCCUPANCY SUMMARY")
            window_order = (
                time_occ_summary_df[
                    ["window_index", "window_label", "window_start", "window_end"]
                ]
                .drop_duplicates()
                .sort_values("window_index")
            )
            for _, window_meta in window_order.iterrows():
                window_label = window_meta["window_label"]
                group = time_occ_summary_df[
                    time_occ_summary_df["window_label"] == window_label
                ]
                print(
                    f"[{int(window_meta['window_start'])},"
                    f"{int(window_meta['window_end'])}): "
                    f"mean OOD="
                    f"{group['mean_candidate_ood_fraction'].mean():.4f} +/- "
                    f"{group['mean_candidate_ood_fraction'].std(ddof=0):.4f} | "
                    f"radius/behavior="
                    f"{group['mean_candidate_radius_ratio_to_behavior'].mean():.3f} | "
                    f"rho(novelty,online)="
                    f"{group['occupancy_vs_online_spearman'].mean():+.4f} +/- "
                    f"{group['occupancy_vs_online_spearman'].std(ddof=0):.4f} | "
                    f"rho(novelty,FQE-overvaluation)="
                    f"{group['occupancy_vs_fqe_rank_overvaluation_spearman'].mean():+.4f} +/- "
                    f"{group['occupancy_vs_fqe_rank_overvaluation_spearman'].std(ddof=0):.4f}"
                )

        if REPLAY_COVERAGE_ABLATION and replayCoverageMetrics:
            coverage_summary_df = pd.DataFrame(replayCoverageMetrics)
            coverage_candidates_df = pd.DataFrame(replayCoverageCandidateRows)

            coverage_summary_df.to_csv(
                f'logs/{DIR}/replay_coverage_ablation.csv',
                index=False,
            )
            coverage_candidates_df.to_csv(
                f'logs/{DIR}/replay_coverage_candidates.csv',
                index=False,
            )

            coverage_order = [
                "full" if w is None else str(int(w))
                for w in REPLAY_COVERAGE_WINDOWS
            ]

            # Compact one-row-per-window summary focused on the hybrid selector.
            hybrid_summary_rows = []
            for coverage_label in coverage_order:
                group = coverage_summary_df[
                    coverage_summary_df["coverage"] == coverage_label
                ]
                if group.empty:
                    continue

                row = {
                    "coverage": coverage_label,
                    "mean_actual_transitions": float(
                        group["actual_transitions"].mean()
                    ),
                    "mean_episodes": float(group["n_episodes"].mean()),
                    "mean_pearson": float(group["pearson"].mean()),
                    "mean_spearman": float(group["spearman"].mean()),
                    "mean_kendall": float(group["kendall"].mean()),
                    "direct_top1_rate": float(
                        group["top1_agreement"].mean()
                    ),
                    "direct_mean_regret": float(
                        group["selection_regret"].mean()
                    ),
                }
                for requested_k in HYBRID_TOPK_VALUES:
                    row[f"oracle_recall_at_{requested_k}"] = float(
                        group[f"oracle_recall_at_{requested_k}"].mean()
                    )
                    row[f"hybrid_mean_regret_at_{requested_k}"] = float(
                        group[f"hybrid_regret_at_{requested_k}"].mean()
                    )
                    row[f"online_reduction_at_{requested_k}"] = float(
                        group[
                            f"hybrid_online_reduction_at_{requested_k}"
                        ].mean()
                    )
                hybrid_summary_rows.append(row)

            pd.DataFrame(hybrid_summary_rows).to_csv(
                f'logs/{DIR}/replay_coverage_hybrid_summary.csv',
                index=False,
            )

            print("---------------------------------")
            print("REPLAY COVERAGE ABLATION SUMMARY")

            for coverage_label in coverage_order:
                group = coverage_summary_df[
                    coverage_summary_df["coverage"] == coverage_label
                ]
                if group.empty:
                    continue

                direct_summary = (
                    f"coverage={coverage_label:>5} | "
                    f"mean actual transitions={group['actual_transitions'].mean():.1f} | "
                    f"mean episodes={group['n_episodes'].mean():.1f} | "
                    f"score_s0={group['score_initial_states'].mean():.1f} | "
                    f"Pearson={group['pearson'].mean():+.4f} "
                    f"+/- {group['pearson'].std(ddof=0):.4f} | "
                    f"Spearman={group['spearman'].mean():+.4f} "
                    f"+/- {group['spearman'].std(ddof=0):.4f} | "
                    f"Kendall={group['kendall'].mean():+.4f} "
                    f"+/- {group['kendall'].std(ddof=0):.4f} | "
                    f"direct_top1={group['top1_agreement'].mean():.3f} | "
                    f"direct_mean_regret={group['selection_regret'].mean():.4f}"
                )

                hybrid_parts = []
                for requested_k in HYBRID_TOPK_VALUES:
                    hybrid_parts.append(
                        f"Recall@{requested_k}="
                        f"{group[f'oracle_recall_at_{requested_k}'].mean():.3f}, "
                        f"HReg@{requested_k}="
                        f"{group[f'hybrid_regret_at_{requested_k}'].mean():.4f}, "
                        f"online_reduction="
                        f"{group[f'hybrid_online_reduction_at_{requested_k}'].mean():.3f}"
                    )

                print(
                    direct_summary
                    + " | "
                    + " | ".join(hybrid_parts)
                )

        np.save(f'logs/{DIR}/distance.npy', distanceArray)
        np.save(f'logs/{DIR}/time.npy', timeArray)
        print("Average distance of random agents to nearest neighbors:", distanceArray)
        print("Time taken for each iteration:", timeArray)

    else:
        for i in range(START_ITER, NUM_ITERS, SEARCH_INTERV):
            print(i)
            model.learn(total_timesteps=SEARCH_INTERV*n_steps_per_rollout*vec_env.num_envs,
                        log_interval=1, 
                        tb_log_name=exp, 
                        reset_num_timesteps=True if i == START_ITER else False, 
                        first_iteration=True if i == START_ITER else False,
                        )

            print_fqe_replay_semantics_stats(model)

            cum_rews = []
            cum_success = []

            if hasattr(args, 'n_envs') and args.n_envs > 1:
                # print("Creating multiple envs - ", args.n_envs)
                # Create a list of environment functions
                dummy_env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=i) for i in range(args.n_envs)]
                dummy_env = SubprocVecEnv(dummy_env_fns)
            else:
                dummy_env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5

                if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                    dummy_env = FlattenObservation(dummy_env)

                dummy_env.reset(seed=args.seed)
            
            if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                mean_rew, std_rew, success = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True, return_success_rate=True)
                print(f'avg 3 return on policy: {mean_rew}')
                print(f'Success rate: {success:.2f}')
                cum_rews.append(mean_rew)
                cum_success.append(success)
            else:
                returns_trains = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True)[0]
                print(f'avg return on 3 trajectories: {returns_trains}')
                cum_rews.append(returns_trains)

            close_env_safely(dummy_env)

            np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
            if env_name in ["FetchReach-v4", "FetchReachDense-v4", "FetchPush-v4", "FetchPushDense-v4"]:
                np.save(f'logs/{DIR}/success_{i}_{i + SEARCH_INTERV}.npy', cum_success)
            timeArray.append(time.time() - start_time)
        
        np.save(f'logs/{DIR}/time.npy', timeArray)
        print("Time taken for each iteration:", timeArray)

    env.close()