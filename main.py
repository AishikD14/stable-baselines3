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

warnings.filterwarnings("ignore")

device = "cpu"

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    arr = np.array(*args)
    return torch.FloatTensor(arr, **kwargs).to(device)

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
def empty_center(data, coor, neighbor, use_momentum, movestep, numiter):
    orig_coor = coor.copy()
    cum_mag = 0
    gamma = 0.9 # discount factor
    momentum = np.zeros(coor.shape)
    es_configs = []
    for i in range(numiter):
        
        distances_, adjs_ = neighbor.kneighbors(coor)
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
def search_empty_space_policies(algo, directory, start, end, env, agent_num=10):
    print("---------------------------------")
    print("Searching empty space policies")

    dt = load_weights(range(start, end), directory, env)
    print(dt.shape)
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(dt)

    _, adjs = neigh.kneighbors(dt[-agent_num:])
    points = dt[adjs[:, 1:]]
    points = points.mean(axis=1)

    policies = []
    for p in points:
        a = empty_center(dt, p.reshape(1, -1), neigh, use_momentum=True, movestep=0.001, numiter=400)
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

# ------------------------------------------------------------------------------------------------------------------------------

env_name = "Ant-v5"
env = gym.make(env_name)
# print(env.action_space, env.observation_space)

n_steps_per_rollout = 200

START_ITER = 1000
SEARCH_INTERV = 1
NUM_ITERS = START_ITER + 10
N_EPOCHS = 10

exp = "PPO_normal_train"
DIR = env_name + "/" + exp + "_" + str(get_latest_run_id('logs/'+env_name+"/", exp)+1)
ckp_dir = f'logs/{DIR}/models'

model = PPO("MlpPolicy", env, verbose=0, seed=0, 
            n_steps=n_steps_per_rollout, 
            batch_size=50, 
            n_epochs=N_EPOCHS, 
            device='cpu', 
            tensorboard_log='logs/'+env_name+"/",
            ckp_dir=ckp_dir)

# print("Starting Initial training")
# model.learn(total_timesteps=START_ITER*n_steps_per_rollout, log_interval=50)
# model.save("full_exp_on_ppo/models/ppo_ant")
# print("Initial training done")
# quit()

print("Loading Initial saved model")

# Load model
model.set_parameters("full_exp_on_ppo/models/ppo_ant", device='cpu')

print("Model loaded")

vec_env = model.get_env()
obs = vec_env.reset()

print("Starting evaluation")

normal_train = False
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
        
        # agents, distance = search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        # agents, distance = neighbor_search_random_walk(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        # agents, distance = random_search_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        # agents, distance = random_search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        agents, distance = random_search_random_walk(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)
        distanceArray.append(distance)
        
        cum_rews = []

        for j, a in enumerate(agents):
            model.policy.load_state_dict(a)
            model.policy.to(device)
            
            returns_trains = evaluate_policy(model, vec_env, n_eval_episodes=5, deterministic=True)[0]
            print(f'avg return on 5 trajectories of agent{j}: {returns_trains}')
            cum_rews.append(returns_trains)
            
        np.save(f'logs/{DIR}/agents_{i}_{i + SEARCH_INTERV}.npy', agents)
        np.save(f'logs/{DIR}/results_{i}_{i + SEARCH_INTERV}.npy', cum_rews)
        timeArray.append(time.time() - start_time)

        best_idx = np.argsort(cum_rews)[-1]
        best_agent = agents[best_idx]
        print(f'the best agent: {best_idx}, avg policy: {cum_rews[best_idx]}')
        load_state_dict(model, best_agent)

    np.save(f'logs/{DIR}/distance.npy', distanceArray)
    np.save(f'logs/{DIR}/time.npy', timeArray)
    print("Average distance of random agents to nearest neighbors:", distanceArray)
    print("Time taken for each iteration:", timeArray)

else:
    for i in range(START_ITER, NUM_ITERS, SEARCH_INTERV):
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