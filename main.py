import copy
import gymnasium as gym
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
# from stable_baselines3.common.fqe import FQE
import torch.nn as nn
import argparse
from data_collection_config import args_ant_dir, args_ant, args_hopper, args_half_cheetah, args_walker2d, args_humanoid, args_cartpole, args_mountain_car, args_pendulum, args_swimmer
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

warnings.filterwarnings("ignore")

device = "cpu"

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    arr = np.array(*args)
    return torch.FloatTensor(arr, **kwargs).to(device)

class ANNAnnoy:
    def __init__(self, dimension, n_neighbors) -> None:
        from annoy import AnnoyIndex
        self.index = AnnoyIndex(dimension, 'euclidean')
        self.index.set_seed(42)
        self.index.build(10)
        self.n_neighbors = n_neighbors
        self.samples = None
    def fit(self, samples):
        for i, s in enumerate(samples):
            self.index.add_item(i, s)
        self.samples = np.concatenate((self.samples, samples)) if self.samples is not None else samples
    def query(self, coor):
        indices,  distances = self.index.get_nns_by_vector(coor[0], self.n_neighbors, include_distances=True)
        return [indices], [distances]
    
class ANNFaiss:
    def __init__(self, dimension, n_neighbors) -> None:
        import faiss
        self.index = faiss.IndexFlatL2(dimension)
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

def load_weights(arng, directory, env, saved_agents):
    policies = []

    for i in range(10):
        policy_vec = []

        # new_model = PPO("MlpPolicy", env, verbose=0, device='cpu')
        # new_model.load(f'logs/{directory}/models/agent{i}.zip', device='cpu')

        if saved_agents:
            ckp = torch.load(f'logs/{directory[:-2]}/models/agent{i+1}.zip', map_location=torch.device('cpu'))
        else:
            ckp = torch.load(f'logs/{directory}/models/agent{i+1}.zip', map_location=torch.device('cpu'))

        # ckp = new_model.policy.state_dict()
        ckp_layers = ckp.keys()

        for layer in ckp_layers:
            if 'value_net' not in layer:
                policy_vec.append(ckp[layer].detach().numpy().reshape(-1))

        policy_vec = np.concatenate(policy_vec)
        policies.append(policy_vec)

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
                policy[layer] = FloatTensor(es_models[i][pivot : pivot + sp].reshape(agent_net[layer].shape))
                pivot += sp
        policies.append(policy)
    return policies

# Nearest neighbor search plus empty space search
def search_empty_space_policies(algo, directory, start, end, env, use_ANN, ANN_lib, saved_agents, agent_num=10):
    print("---------------------------------")
    print("Searching empty space policies")

    dt = load_weights(range(start, end), directory, env, saved_agents)
    print(dt.shape)

    if not use_ANN:
        # Calculate the nearest neighbors of the agents using KNN
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
    points = points.mean(axis=1)

    # Choose every second point
    points = points[::2]

    policies = []
    for p in points:
        a = empty_center(dt, p.reshape(1, -1), neigh, use_ANN, use_momentum=True, movestep=0.001, numiter=60)
        policies.append(a[1])
    policies = np.concatenate(policies)
    print(policies.shape)

    agents = dump_weights(algo.policy.state_dict(), policies)

    if not use_ANN:
        # Calculate the nearest neighbors of the agents using KNN
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
    points = points.mean(axis=1)
    print(points.shape)

    # Calculate the distance of each agent to the mean of the nearest neighbors
    distances = [euclidean(policy, point) for policy, point in zip(policies, points)]

    # Average distance
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
def neighbor_search_random_walk(algo, directory, start, end, env, agent_num=10):
    print("---------------------------------")
    print("Searching random policies")

    dt = load_weights(range(start, end), directory, env)
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
        a = empty_center(dt, p.reshape(1, -1), neigh, use_momentum=True, movestep=0.001, numiter=400)
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
    fqe = FQE(model.replay_buffer.obs_shape[0], model.replay_buffer.action_dim, lr=1e-4, gamma=model.gamma, device='cuda:0')
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
        """Directly use PPO's policy without relying on _impl."""
        if isinstance(x, (list, tuple)):  # Handle complex observations
            x = [xi.to(self.device) for xi in x]
        else:
            x = x.to(self.device)
        
        # Convert to numpy for SB3 compatibility
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        actions, _ = self.ppo.predict(x_np, deterministic=True)
        return torch.as_tensor(actions, device=self.device)

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

# Method for evaluating the FQE using d3rlpy
def d3rl_evaluation(model, exp_name):
    # terminals = model.replay_buffer.dones

    # print("[INFO] Forcing terminal flags every", args.n_steps_per_rollout, "steps.")
    # terminals = np.zeros_like(model.replay_buffer.rewards, dtype=bool)
    # terminals[args.n_steps_per_rollout - 1 :: args.n_steps_per_rollout] = True

    # Flatten the replay buffer data
    observations = model.replay_buffer.observations.reshape(-1, model.replay_buffer.observations.shape[-1])
    actions = model.replay_buffer.actions.reshape(-1, model.replay_buffer.actions.shape[-1])
    rewards = model.replay_buffer.rewards.reshape(-1, 1)
    terminals = model.replay_buffer.dones.reshape(-1, 1)
    # terminals =terminals.reshape(-1, terminals.shape[-1])

    # print("Observations shape:", observations.shape)
    # print("Actions shape:", actions.shape)
    # print("Rewards shape:", rewards.shape)
    # print("Terminals shape:", terminals.shape)

    # Build and return the MDPDataset
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )

    try:
        ppo_wrapper = PPOQWrapper(model)

        # 4. Build the wrapper internals using dataset info
        ppo_wrapper.build_with_dataset(dataset)

    except Exception as e:
        print(f"Error creating or building the PPOQWrapper: {e}")
        import traceback
        traceback.print_exc()
        exit()

    try:
        fqe = d3rlpy.ope.FQE(
            algo=ppo_wrapper, # Pass the wrapper instance
            config=d3rlpy.ope.FQEConfig(
                learning_rate=3e-4,         # Learning rate for FQE's internal Q-network
                target_update_interval=100, # How often to update FQE's target network
                gamma=ppo_wrapper.ppo.gamma, # Discount factor
            )
        )

        print("--------------------------------------------------------------------------------")
        print("Fitting d3rlpy FQE...")

        # Consider using a smaller number of steps for initial testing
        N_STEPS = 10000 # 10000
        N_STEPS_PER_EPOCH = 1000 # 1000

        output = fqe.fit(
            dataset,
            n_steps=N_STEPS,
            n_steps_per_epoch=N_STEPS_PER_EPOCH,
            evaluators={
                # Estimates the expected value of the initial states according to FQE's learned Q-function
                'init_value': d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
                # Soft Off-Policy Classification: Measures if the policy achieves a certain return threshold
                # 'soft_opc': d3rlpy.metrics.SoftOPCEvaluator(return_threshold=1000), # Adjust threshold based on env/task
            },
            show_progress=True,
            save_interval=N_STEPS//N_STEPS_PER_EPOCH,
            experiment_name=exp_name
        )

        print("\nFQE Fitting completed.")

        # The primary result is often the initial state value estimate
        initial_state_value = output[-1][1]['init_value'] # Get from the last epoch's results
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
                    policy[layer] = torch.FloatTensor(theta[pivot:pivot + sp].reshape(policy_state[layer].shape))
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
            new_policy[layer] = torch.FloatTensor(theta_new[pivot:pivot + sp].reshape(policy_state[layer].shape))
            pivot += sp

    agent_list = [new_policy]  # we return one updated agent
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

    return agent_list, 0.0

# ------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

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

    # ------------------------------------------------------------------------------------------------------------

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
        # Create a list of environment functions
        env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=i) for i in range(args.n_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5
        env.reset(seed=args.seed)
        
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

    exp = "PPO_PBT"
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

    START_ITER = 1000000 // (args.n_steps_per_rollout*args.n_envs)
    SEARCH_INTERV = 1 # Make this 2 for n_epochs=5 and keep 1 for n_epochs=10
    NUM_ITERS = 3000000 // (args.n_steps_per_rollout*args.n_envs)
    N_EPOCHS = args.n_epochs

    # ---------------------------------------------------------------------------------------------------------------

    # print("Starting Initial training")
    # os.makedirs(f'full_exp_on_ppo/models/'+env_name, exist_ok=True)

    # model.learn(total_timesteps=1000000, log_interval=50, tb_log_name=exp, init_call=True)
    # model.save("full_exp_on_ppo/models/"+env_name+"/ppo_hopper_1M"+'_'+str(args.seed))

    # print("Initial training done") 

    # print("Saving replay buffer for later use")
    # os.makedirs(f'full_exp_on_ppo/replay_buffers/'+env_name, exist_ok=True)

    # # Save the replay buffer
    # np.savez(f'full_exp_on_ppo/replay_buffers/'+env_name+'/replay_buffer_'+str(args.seed)+'.npz',
    #     observations=model.replay_buffer.observations.reshape(-1, model.replay_buffer.observations.shape[-1]),
    #     actions=model.replay_buffer.actions.reshape(-1, model.replay_buffer.actions.shape[-1]),
    #     rewards=model.replay_buffer.rewards.reshape(-1, model.replay_buffer.rewards.shape[-1]),
    #     terminals=model.replay_buffer.dones.reshape(-1, model.replay_buffer.dones.shape[-1])
    # )
    
    # print("Replay buffer saved")

    # quit()

    # ----------------------------------------------------------------------------------------------------------------

    print("Loading Initial saved model")

    model.set_parameters(args.init_model_path+'_'+str(args.seed), device=args.device)

    print("Model loaded")

    # -------------------------------------------------------------------------------------------------------------

    # print("Loading replay buffer")

    # # Load the replay buffer
    # replay_buffer = np.load(f'full_exp_on_ppo/replay_buffers/'+env_name+'/replay_buffer_'+str(args.seed)+'.npz')
    # model.replay_buffer.observations = replay_buffer['observations'] if args.n_envs == 1 else replay_buffer['observations'].reshape(-1, args.n_envs, replay_buffer['observations'].shape[-1])
    # model.replay_buffer.actions = replay_buffer['actions'] if args.n_envs == 1 else replay_buffer['actions'].reshape(-1, args.n_envs, replay_buffer['actions'].shape[-1])
    # model.replay_buffer.rewards = replay_buffer['rewards'] if args.n_envs == 1 else replay_buffer['rewards'].reshape(-1, args.n_envs)
    # model.replay_buffer.dones = replay_buffer['terminals'] if args.n_envs == 1 else replay_buffer['terminals'].reshape(-1, args.n_envs)
    # print("Replay buffer loaded")
    # print("Replay buffer shape: ", model.replay_buffer.observations.shape, model.replay_buffer.actions.shape, model.replay_buffer.rewards.shape, model.replay_buffer.dones.shape)

    # ----------------------------------------------------------------------------------------------------------------

    vec_env = model.get_env()
    obs = vec_env.reset()

    print("Starting evaluation")

    normal_train = False
    use_ANN = False
    ANN_lib = "Annoy"
    online_eval = True

    saved_agents = False
    saved_iter = 5823
    model_already_learned = True

    distanceArray = []
    start_time = time.time()
    timeArray = []

    avg_checkpoint = False
    use_ptb = True

    if exp == "PPO_baseline":
        # START_ITER = 1953
        # NUM_ITERS = 9765

        # For Pendulum-v1
        START_ITER = 976
        NUM_ITERS = 2930

    if saved_agents:
        # Find best agent index
        best_agent_index = np.load(f'logs/{env_name}/{exp}/best_agent_{str(saved_iter-SEARCH_INTERV)}_{str(saved_iter)}.npy')
        print("Last Best agent index: ", best_agent_index[0])

        ckp = torch.load(f'logs/{env_name}/{exp}/models/agent{str(best_agent_index[0])}.zip', map_location=torch.device('cpu'))
        print("Checkpoint loaded")

        load_state_dict(model, ckp)
        print("Model loaded")

        START_ITER = saved_iter

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

            if not avg_checkpoint and not use_ptb:
                agents, distance = search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env, use_ANN, ANN_lib, saved_agents and model_already_learned)
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

                os.makedirs(f'logs/{DIR}', exist_ok=True)
                np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents)
                np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
                timeArray.append(time.time() - start_time)

                load_state_dict(model, agents[0])

                continue

            # -----------------------------------------------------------------------------------

            if use_ptb:
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
                    dummy_env.reset(seed=args.seed)

                returns_trains = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True)[0]
                print(f'avg return on 3 trajectories of agent{j}: {returns_trains}')
                cum_rews.append(returns_trains)

                # Q-function evaluation
                if not online_eval:
                    # Advantage estimation code
                    # q_adv, q_loss = advantage_evaluation(model, args)
                    # advantage_rew.append(q_adv)
                    # q_losses.append(q_loss)

                    # d3rl FQE evaluation code
                    init_est = d3rl_evaluation(model, f"{'-'.join(DIR.split('/'))}")
                    advantage_rew.append(init_est)

            if not online_eval:
                # print(f'ave q losses: {np.mean(q_losses)}, std: {np.std(q_losses)}')
                print(f'ave advantage rew: {np.mean(advantage_rew)}, std: {np.std(advantage_rew)}')
            print(f'avg cum rews: {np.mean(cum_rews)}, std: {np.std(cum_rews)}')    

            os.makedirs(f'logs/{DIR}', exist_ok=True)

            np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents)
            if online_eval:
                np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
            if not online_eval:
                np.save(f'logs/{DIR}/adv_results_{i}_{i + SEARCH_INTERV}.npy', advantage_rew)
            timeArray.append(time.time() - start_time)

            # Correlation calculation
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
                best_idx = np.argsort(cum_rews)[-1]
                best_agent = agents[best_idx]
                print(f'the best agent: {best_idx}, best agent cum rewards: {cum_rews[best_idx]}')
                best_agent_index.append(best_idx)
                np.save(f'logs/{DIR}/best_agent_{i}_{i + SEARCH_INTERV}.npy', best_agent_index)
                load_state_dict(model, best_agent)

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

            cum_rews = []

            if hasattr(args, 'n_envs') and args.n_envs > 1:
                # print("Creating multiple envs - ", args.n_envs)
                # Create a list of environment functions
                dummy_env_fns = [make_envs(env_name, seed=args.seed)(seed_offset=i) for i in range(args.n_envs)]
                dummy_env = SubprocVecEnv(dummy_env_fns)
            else:
                dummy_env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5
                dummy_env.reset(seed=args.seed)
            # dummy_env = gym.make(env_name)
            # dummy_env.reset(seed=args.seed)
            returns_trains = evaluate_policy(model, dummy_env, n_eval_episodes=3, deterministic=True)[0]
            print(f'avg 3 return on policy: {returns_trains}')
            cum_rews.append(returns_trains)
            np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
            timeArray.append(time.time() - start_time)
        
        np.save(f'logs/{DIR}/time.npy', timeArray)
        print("Time taken for each iteration:", timeArray)

    env.close()