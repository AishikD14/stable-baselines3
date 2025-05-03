import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from collections import OrderedDict
import torch
from torch.optim import Adam
import time
import os
from stable_baselines3.common.utils import get_latest_run_id, safe_mean
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
from environments.make_env import make_env
import pandas as pd
from stable_baselines3.common.fqe import FQE
import torch.nn as nn
import argparse
from data_collection_config import args_ant_dir, args_ant, args_hopper, args_half_cheetah, args_walker2d, args_humanoid
from d3rlpy.algos import BCConfig
import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.algos import QLearningAlgoBase
from d3rlpy.base import LearnableConfig
from d3rlpy.constants import ActionSpace
import matplotlib.pyplot as plt

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
    orig_coor = coor.copy()
    cum_mag = 0
    gamma = 0.9  # discount factor
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
            if use_momentum:
                rw_configs.extend(coor.tolist())

    rw_configs.extend(coor.tolist())

    if not use_momentum:
        return np.linalg.norm(coor - orig_coor), coor
    else:
        return np.linalg.norm(coor - orig_coor), np.array(rw_configs)

def load_weights(arng, directory, env):
    policies = []

    for i in range(10):
        policy_vec = []

        # new_model = PPO("MlpPolicy", env, verbose=0, device='cpu')
        # new_model.load(f'logs/{directory}/models/agent{i}.zip', device='cpu')

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
def search_empty_space_policies(algo, directory, start, end, env, use_ANN, ANN_lib, agent_num=10):
    print("---------------------------------")
    print("Searching empty space policies")

    dt = load_weights(range(start, end), directory, env)
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
    dt = load_weights(range(start, end), directory, env)
    print(dt.shape)
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(dt)

    _, adjs = neigh.kneighbors(dt[-agent_num:])
    points = dt[adjs[:, 1:]]
    points = points.mean(axis=1)
    print(points.shape)

    policies = []
    print(len(points))
    for p in points:
        a = random_walk(dt, p.reshape(1, -1), neigh, use_momentum=True, movestep=0.001, numiter=400)
        policies.append(a[1])
    policies = np.concatenate(policies)
    print(policies.shape)

    agents = dump_weights(algo.policy.state_dict(), policies)

    # Calculate the nearest neighbors of the agents
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(policies)

    _, adjs = neigh.kneighbors(policies)
    points = policies[adjs[:, 1:]]
    points = points.mean(axis=1)
    print(points.shape)

    # Calculate the distance of each agent to the mean of the nearest neighbors
    distances = [euclidean(policy, point) for policy, point in zip(policies, points)]

    # Average distance
    average_distance = np.mean(distances)
    print("Average distance of agents to nearest neighbors:", average_distance)
    
    return agents, average_distance

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
def advantage_evaluation(model, horizon=1000):
    fqe = FQE(model.replay_buffer.obs_shape[0], model.replay_buffer.action_dim, lr=1e-4, gamma=model.gamma, device='cuda:0')
    fqe.build_q_net(model)
    q_loss = fqe.train(fqe.model, 256, 10)
    s0 = [model.env.reset() for _ in range(100)]
    s0 = FloatTensor(s0)
    s0 = s0.squeeze(1)
    act, _, _ = model.policy(s0)
    q_pred = fqe.predict(s0, act.detach(), horizon=horizon)
    # print(f'q_pred: {q_pred}, q_loss: {q_loss}')
    return q_pred, q_loss

def d3rl_evaluation(model):
    print(model.replay_buffer.observations.shape)
    print(model.replay_buffer.actions.shape)
    print(model.replay_buffer.rewards.shape)
    print(model.replay_buffer.dones.shape)

    # Flatten the replay buffer data
    # observations = model.replay_buffer.observations.reshape(-1, model.replay_buffer.observations.shape[-1])
    # actions = model.replay_buffer.actions.reshape(-1, model.replay_buffer.actions.shape[-1])
    # rewards = model.replay_buffer.rewards.reshape(-1, model.replay_buffer.rewards.shape[-1])
    # dones = model.replay_buffer.dones.reshape(-1, model.replay_buffer.dones.shape[-1])

    # print("Observations shape:", observations.shape)
    # print("Actions shape:", actions.shape)
    # print("Rewards shape:", rewards.shape)
    # print("Terminals shape:", dones.shape)

    # quit()

    # terminals = model.replay_buffer.dones

    # Check if either terminals or timeouts contain any True value
    # if not (np.any(terminals) or np.any(timeouts)):
    print("[INFO] Forcing terminal flags every", args.n_steps_per_rollout, "steps.")
    terminals = np.zeros_like(model.replay_buffer.rewards, dtype=bool)
    terminals[args.n_steps_per_rollout - 1 :: args.n_steps_per_rollout] = True

    # Flatten the replay buffer data
    observations = model.replay_buffer.observations.reshape(-1, model.replay_buffer.observations.shape[-1])
    actions = model.replay_buffer.actions.reshape(-1, model.replay_buffer.actions.shape[-1])
    rewards = model.replay_buffer.rewards.reshape(-1, model.replay_buffer.rewards.shape[-1])
    terminals =terminals.reshape(-1, terminals.shape[-1])

    print("Observations shape:", observations.shape)
    print("Actions shape:", actions.shape)
    print("Rewards shape:", rewards.shape)
    print("Terminals shape:", terminals.shape)

    # Build and return the MDPDataset
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )

    dataset.dump("ppo_ant_d3rlpy_buffer.h5")

    print("Observation shape:", dataset.episodes[0])
    quit()

    # Split into training and testing episodes
    train_episodes, test_episodes = train_test_split(dataset.episodes, test_size=0.2)

    # Create BC (continuous) with config
    bc_config = BCConfig()
    bc = bc_config.create(device='cuda')  # or 'cpu' if no GPU

    # Train BC model
    bc.build_with_dataset(dataset)
    bc.fit(train_episodes, n_epochs=10)

    # Run FQE
    fqe = d3rlpy.ope.FQE(algo=bc, device='cuda')  # or 'cpu'
    fqe.build_with_dataset(dataset)
    fqe.fit(train_episodes, n_epochs=10)

    # Estimate return of the learned BC policy
    estimated_return = fqe.predict_value(test_episodes)
    print(f"Estimated return via FQE: {estimated_return:.2f}")

    quit()

# ------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

env_name = "Ant-v5" # For standard ant locomotion task (single goal task)
# env_name = "HalfCheetah-v5" # For standard half-cheetah locomotion task (single goal task)
# env_name = "Hopper-v5" # For standard hopper locomotion task (single goal task)
# env_name = "Walker2d-v5" # For standard walker locomotion task (single goal task)
# env_name = "Humanoid-v5" # For standard ant locomotion task (single goal task)

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

env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5
# env = make_env(env_name, episodes_per_task=1, seed=0, n_tasks=1) # For AntDir-v0

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

START_ITER = 1000000 // args.n_steps_per_rollout
SEARCH_INTERV = 2 # Since PPO make n_epochs=10 updates with each rollout, we can set this to 1 instead of 10
NUM_ITERS = 3000000 // args.n_steps_per_rollout
N_EPOCHS = args.n_epochs

# ---------------------------------------------------------------------------------------------------------------

exp = "PPO_test"
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
    batch_size=args.batch_size,
    gamma=args.gamma,
    ent_coef=args.ent_coef,
    learning_rate=args.learning_rate,
    clip_range=args.clip_range,
    max_grad_norm=args.max_grad_norm,
    n_epochs=args.n_epochs,
    gae_lambda=args.gae_lambda,
    vf_coef=args.vf_coef,
    device=args.device,
    tensorboard_log=args.tensorboard_log,
    ckp_dir=ckp_dir
)

if policy_kwargs:
    ppo_kwargs["policy_kwargs"] = policy_kwargs

if normalize_kwargs:
    ppo_kwargs["normalize_kwargs"] = normalize_kwargs

model = PPO(**ppo_kwargs)

# ---------------------------------------------------------------------------------------------------------------

# print("Starting Initial training")
# model.learn(total_timesteps=1000000, log_interval=50, tb_log_name=exp, init_call=True)
# model.save("full_exp_on_ppo/models/"+env_name+"/ppo_hopper_1M_1")
# print("Initial training done") 
# quit()

# ----------------------------------------------------------------------------------------------------------------

print("Loading Initial saved model")

model.set_parameters(args.init_model_path, device=args.device)

print("Model loaded")

# -------------------------------------------------------------------------------------------------------------

vec_env = model.get_env()
obs = vec_env.reset()

# model.learn(total_timesteps=1000, log_interval=50, tb_log_name=exp, init_call=True)

print("Starting evaluation")

# ------------------------------------------------------------------------------------------------------------------

# Define a minimal nn.Module to satisfy base class requirements for _impl
class DummyImpl(torch.nn.Module):
    """
    A minimal placeholder implementation module required by QLearningAlgoBase.
    FQE uses its own implementation but the base class needs this structure.
    Methods here might be called during initialization or by base class logic,
    but FQE's core fitting loop relies on the wrapper's predict_* methods.
    """
    def __init__(self, observation_shape, action_size, device):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.device = device
        # Add a dummy parameter to ensure the module has parameters and can be moved to device
        # This helps avoid potential issues with device placement checks.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        # Return dummy zeros with the expected batch dimension and Q-value shape (batch, 1)
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 1), device=self.device)

    def predict_value(self, x: torch.Tensor, action: torch.Tensor, with_std: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Return dummy values if called directly on impl
        batch_size = x.shape[0]
        zeros = torch.zeros((batch_size, 1), device=self.device)
        if with_std:
            return zeros, zeros
        else:
            return zeros

    def predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # Return dummy zeros with appropriate action shape.
        batch_size = x.shape[0]
        # For continuous action space, action_size is the dimension
        # For discrete, it would be the number of actions (not handled here)
        action_shape = (batch_size, self.action_size) if isinstance(self.action_size, int) else (batch_size,) + tuple(self.action_size) # Handle tuple shapes
        return torch.zeros(action_shape, device=self.device)

class PPOQWrapper(QLearningAlgoBase):
    def __init__(self, ppo_policy):
        super().__init__(
            config=LearnableConfig(),
            device=str(ppo_policy.device),
            enable_ddp=False
        )
        self.ppo = ppo_policy
        self._action_size = None # To store action size/shape
        self._observation_shape = None # To store observation shape
        print(f"PPOQWrapper initialized on device: {self.device}")

    def get_action_type(self):
        return ActionSpace.CONTINUOUS  # For PPO in continuous control tasks
        
    # 1. Mandatory method for network initialization
    def inner_create_impl(self, observation_shape, action_size):
        # class PPOTwinQ(torch.nn.Module):
        #     def __init__(self, ppo):
        #         super().__init__()
        #         self.ppo = ppo
                
        #     def forward(self, x, action=None):
        #         with torch.no_grad():
        #             return self.ppo.critic(x).reshape(-1, 1)
                    
        # self._impl = PPOTwinQ(self.ppo)

        # Minimal implementation to satisfy FQE's checks
        # class DummyImpl(torch.nn.Module):
        #     def forward(self, x, action=None):
        #         return torch.zeros(x.shape[0], 1)  # Placeholder
        
        # self._impl = DummyImpl()

        self._impl = self.ppo.policy  # Use PPO's policy directly

        # Simplified Q-network that delegates to PPO's actor/critic
        # class FQECompatQ(torch.nn.Module):
        #     def __init__(self, wrapper):
        #         super().__init__()
        #         self.wrapper = wrapper
                
        #     def predict_best_action(self, x):
        #         return self.wrapper.predict(x)
                
        #     def forward(self, x, action=None):
        #         # Use PPO's critic for Q-values
        #         with torch.no_grad():
        #             return self.wrapper.ppo.critic(x).reshape(-1, 1)
        
        # self._impl = FQECompatQ(self)
        
    # 2. Required for dataset compatibility
    def build_with_dataset(self, dataset):
        self.create_impl(dataset.episodes[0].serialize()['observations'].shape, dataset.episodes[0].serialize()['actions'].shape)

    # 3. Action prediction method
    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Stable-Baselines3's official prediction interface
        action, _ = self.ppo.predict(x, deterministic=True)
        return action
        
    def predict_best_action(self, x):
        return self.predict(x)  # For policy-based algorithms, best=current

# load from HDF5
with open("ppo_ant_d3rlpy_buffer.h5", "rb") as f:
    dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

# print("Observation shape:", dataset.episodes[0].serialize()['observations'].shape)
# print("Action shape:", dataset.episodes[0].serialize()['actions'].shape)
# print("Reward shape:", dataset.episodes[0].serialize()['rewards'].shape)
# print("Terminal shape:", dataset.episodes[0].serialize()['terminated'].shape)

# Create FQE-compatible wrapper
ppo_wrapper = PPOQWrapper(model)
ppo_wrapper.build_with_dataset(dataset) 

# Split into training and testing episodes
# train_episodes, test_episodes = train_test_split(dataset.episodes, test_size=0.2)

# print("--------------------------------------------------------------------------------")
# print("Initializing d3rlpy BC")

# # Create BC (continuous) with config
# bc_config = BCConfig(encoder_factory=VectorEncoderFactory())
# bc = bc_config.create(device='cpu')  # or 'cpu' if no GPU

# # Train BC model
# bc.build_with_dataset(dataset)

# print("--------------------------------------------------------------------------------")

# bc.fit(
#     dataset,
#     n_steps=100,  # Typically needs more epochs than 10
# )

print("--------------------------------------------------------------------------------")
print("Initialzing d3rlpy FQE")

# Run FQE
fqe = d3rlpy.ope.FQE(
        algo=ppo_wrapper, 
        config=d3rlpy.ope.FQEConfig(
            learning_rate=3e-4,
            target_update_interval=100
        )
    )

print("--------------------------------------------------------------------------------")
print("Fitting d3rlpy FQE")

output = fqe.fit(dataset, 
        n_steps=10000,
        n_steps_per_epoch=1000,
        evaluators={
            'init_value': d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
            'soft_opc': d3rlpy.metrics.SoftOPCEvaluator(return_threshold=-300),
        },
        show_progress=True,
    )

print("--------------------------------------------------------------------------------")
print("FQE output: ", output)
# print("FQE output: ", output['init_value'])
# print("FQE output: ", output['soft_opc'])

print("--------------------------------------------------------------------------------")
# print("Estimating return")

# Estimate return of the learned BC policy
# estimated_return = fqe.predict_value(dataset.episodes[0].serialize()['observations'], dataset.episodes[0].serialize()['actions'])

# print("---------------------------------------------------------------------------------")
# print("Estimated return via FQE: ", estimated_return)

# print("Estimated return via FQE: ", estimated_return.mean())

# print("Estimated return via FQE: ", estimated_return.std())

# Plot the estimated return
# plt.plot(estimated_return)
# plt.title("Estimated Return via FQE")
# plt.xlabel("Time Step")
# plt.ylabel("Estimated Return")
# plt.grid()
# plt.show()

quit()

# -------------------------------------------------------------------------------------------------------------------

normal_train = False
use_ANN = False
ANN_lib = "Annoy"
online_eval = False

distanceArray = []
start_time = time.time()
timeArray = []

if exp == "PPO_baseline":
    START_ITER = 1953
    NUM_ITERS = 9765

if not normal_train:
    for i in range(START_ITER, NUM_ITERS, SEARCH_INTERV):
        print(i)
        model.learn(total_timesteps=SEARCH_INTERV*n_steps_per_rollout*vec_env.num_envs,
                    log_interval=1, 
                    tb_log_name=exp, 
                    reset_num_timesteps=True if i == START_ITER else False, 
                    first_iteration=True if i == START_ITER else False,
                    )
        
        agents, distance = search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env, use_ANN, ANN_lib)
        # agents, distance = neighbor_search_random_walk(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        # agents, distance = random_search_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        # agents, distance = random_search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        # agents, distance = random_search_random_walk(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        distanceArray.append(distance)
        
        cum_rews = []
        best_agent_index = []
        advantage_rew = []
        q_losses = []

        for j, a in enumerate(agents):
            model.policy.load_state_dict(a)
            model.policy.to(device)
            
            # Online evaluation
            # returns_trains = evaluate_policy(model, vec_env, n_eval_episodes=3, deterministic=True)[0]
            # print(f'avg return on 5 trajectories of agent{j}: {returns_trains}')
            # cum_rews.append(returns_trains)

            # Q-function evaluation
            if not online_eval:
                # q_adv, q_loss = advantage_evaluation(model)
                # advantage_rew.append(q_adv)
                # q_losses.append(q_loss)

                d3rl_evaluation(model)

        if not online_eval:
            print(f'ave q losses: {np.mean(q_losses)}, std: {np.std(q_losses)}')
            print(f'ave advantage rew: {np.mean(advantage_rew)}, std: {np.std(advantage_rew)}')
        print(f'ave cum rews: {np.mean(cum_rews)}, std: {np.std(cum_rews)}')    

        np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents)
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
            print("Pearson correlation coefficient:", corr_pear['advantage'][1])
            print("Spearman correlation coefficient:", corr_spearman['advantage'][1])

            # Using the best agent from the top 5
            top_5_idx = np.argsort(advantage_rew)[-5:]
            top_5_agents = np.array(agents)[top_5_idx]
            best_agent, best_idx, returns_trains = None, None, -float('inf')
            dummy_env = gym.make(env_name)
            for idx, tagent in enumerate(top_5_agents):
                model.policy.load_state_dict(tagent)
                model.policy.to(device)
                cur_return = evaluate_policy(model, dummy_env, n_eval_episodes=2, callback=evaluation_callback, deterministic=True)[0]
                if cur_return > returns_trains:
                    returns_trains = cur_return
                    best_idx = idx
            best_agent = top_5_agents[best_idx]

            print(f'the best agent: {best_idx}, avg policy: {returns_trains}')
            best_agent_index.append(best_idx)
            np.save(f'logs/{DIR}/best_agent_{i}_{i + SEARCH_INTERV}.npy', best_agent_index)
            load_state_dict(model, best_agent)

        # Finding the best agent from online evaluation
        if online_eval:
            best_idx = np.argsort(cum_rews)[-1]
            best_agent = agents[best_idx]
            print(f'the best agent: {best_idx}, avg policy: {cum_rews[best_idx]}')
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

        returns_trains = evaluate_policy(model, vec_env, n_eval_episodes=3, deterministic=True)[0]
        print(f'avg return on policy: {returns_trains}')
        cum_rews.append(returns_trains)
        np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
        timeArray.append(time.time() - start_time)
    
    np.save(f'logs/{DIR}/time.npy', timeArray)
    print("Time taken for each iteration:", timeArray)

env.close()