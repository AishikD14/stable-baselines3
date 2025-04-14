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
    gamma = 0.9 # discount factor
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

    for i in range(N_EPOCHS):
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
    # points = points[::2]

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

# Function to evaluate the advantage of the policy
def advantage_evaluation(model):
    iss = []
    with torch.no_grad():
        for rollout_data in model.rollout_buffer.get(model.batch_size):
            actions = rollout_data.actions

            values, log_prob, entropy = model.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            # -importance_sampling = advantages * ratio
            # iss.append(importance_sampling.sum().item())
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 0.9, 1.1)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
            iss.append(policy_loss.item())
    iss = np.sum(iss)
    return iss

# ------------------------------------------------------------------------------------------------------------------------------

env_name = "Ant-v5" # For standard ant locomotion task (single goal task)
# env_name = "AntDir-v0" # Part of the Meta-World or Meta-RL (meta-reinforcement learning) benchmarks (used for multi-task learning)

env = gym.make(env_name) # For Ant-v5
# env = make_env(env_name, episodes_per_task=1, seed=0, n_tasks=1) # For AntDir-v0

print("Environment created")
print(env.action_space, env.observation_space)
# ------------------------------------------------------------------------------------------------------------
# goal = np.random.uniform(0, 3.1416)
# env = gym.make(env_name, goal=goal) # multi-task learning

# print(env.action_space, env.observation_space)

n_steps_per_rollout = 200

# --------------------------------------------------------------------------------------------------------------

# START_ITER = 1000   #For 200k steps initialisation (Normal hyperparameters)
START_ITER = 5000   #For 1M steps initialisation (Optimal hyperparameters)
# START_ITER = 25000  #For 5M steps initialisation (Just used for visualization right now)

SEARCH_INTERV = 1 # Since PPO make n_epochs=10 updates with each rollout, we can set this to 1 instead of 10

# NUM_ITERS = START_ITER + 100 # Just for testing
NUM_ITERS = START_ITER + 20000 #5M steps

N_EPOCHS = 10 # Since set to 10 updates per rollout

# ---------------------------------------------------------------------------------------------------------------

exp = "PPO_empty_space_ls"
DIR = env_name + "/" + exp + "_" + str(get_latest_run_id('logs/'+env_name+"/", exp)+1)
ckp_dir = f'logs/{DIR}/models'

# Normal hyperparameters
# model = PPO("MlpPolicy", env, verbose=0, seed=0, 
#             n_steps=n_steps_per_rollout, 
#             batch_size=50, 
#             n_epochs=N_EPOCHS, 
#             device='cpu', 
#             tensorboard_log='logs/'+env_name+"/",
#             ckp_dir=ckp_dir)

# Best hyperparameters
model = PPO("MlpPolicy", env, verbose=0, seed=0, 
                n_steps=512, 
                # n_steps=200,
                batch_size=32, 
                gamma=0.98,
                ent_coef=4.9646e-07,
                learning_rate=1.90609e-05,
                clip_range=0.1,
                max_grad_norm=0.6,
                n_epochs=10,
                gae_lambda=0.8,
                vf_coef=0.677239,
                device=device, 
                tensorboard_log='logs/'+env_name+"/",
                ckp_dir=ckp_dir)

# print("Starting Initial training")
# model.learn(total_timesteps=START_ITER*n_steps_per_rollout, log_interval=50, tb_log_name=exp)
# model.save("full_exp_on_ppo/models/ppo_ant_5M_1")
# print("Initial training done") 
# quit()

print("Loading Initial saved model")

# Load model
# model.set_parameters("full_exp_on_ppo/models/ppo_ant_200k", device='cpu') # Normal hyperparameters for Ant
model.set_parameters("full_exp_on_ppo/models/ppo_ant_1M", device='cpu') # Best hyperparameters for Ant
# model.set_parameters("full_exp_on_ppo/models/ppo_antdir_1M", device='cpu') # Best hyperparameters for Antdir

print("Model loaded")

vec_env = model.get_env()
obs = vec_env.reset()

print("Starting evaluation")

normal_train = False
use_ANN = False
ANN_lib = "Annoy"

distanceArray = []
start_time = time.time()
timeArray = []

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

        for j, a in enumerate(agents):
            model.policy.load_state_dict(a)
            model.policy.to(device)
            
            returns_trains = evaluate_policy(model, vec_env, n_eval_episodes=3, deterministic=True)[0]
            print(f'avg return on 5 trajectories of agent{j}: {returns_trains}')
            cum_rews.append(returns_trains)
            advantage_rew.append(advantage_evaluation(model))
            
        np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents)
        np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
        timeArray.append(time.time() - start_time)

        df = pd.DataFrame({
            'advantage': advantage_rew,
            'online': cum_rews
        })
        corr_pear = df.corr(method='pearson')
        corr_spearman = df.corr(method='spearman')
        print("Pearson correlation coefficient:", corr_pear['advantage'][1])
        print("Spearman correlation coefficient:", corr_spearman['advantage'][1])

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

        returns_trains = evaluate_policy(model, vec_env, n_eval_episodes=5, deterministic=True)[0]
        print(f'avg return on policy: {returns_trains}')
        cum_rews.append(returns_trains)
        np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
        timeArray.append(time.time() - start_time)
    
    np.save(f'logs/{DIR}/time.npy', timeArray)
    print("Time taken for each iteration:", timeArray)

env.close()