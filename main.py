import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from collections import OrderedDict
import torch
import time
from stable_baselines3.common.utils import get_latest_run_id
# from stable_baselines3.common.base_class import BaseAlgorithm

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
    print("-------------------")
    orig_coor = coor.copy()
    cum_mag = 0
    gamma = 0.9 # discount factor
    momentum = np.zeros(coor.shape)
    es_configs = []
    for i in range(numiter):
        print(i)
        contains_nan = np.isnan(coor).any()
        print("Contains NaN:", contains_nan)
        print(coor)
        print("-------------------")
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

def load_weights(arng, directory, env):
    policies = []
    for i in arng:
        policy_vec = []

        new_model = PPO("MlpPolicy", env, verbose=0, device='cpu')
        new_model.load(f'logs/{directory}/models/agent{i}.zip', device='cpu')

        # print(new_model.policy)

        ckp = new_model.policy.state_dict()
        
        ckp_layers = ckp.keys()
        for layer in ckp_layers:
            if 'policy' in layer:
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
            # if args.policy == 'ppo_actor':
            #     if 'actor_mean' not in layer:
            #         policy[layer] = agent_net[layer]
            #     else:
            #         sp = agent_net[layer].reshape(-1).shape[0]
            #         policy[layer] = ptu.FloatTensor(es_models[i][pivot : pivot + sp].reshape(agent_net[layer].shape))
            #         pivot += sp
            # else:
            if 'policy' not in layer:
                policy[layer] = agent_net[layer]
            else:
                sp = agent_net[layer].reshape(-1).shape[0]
                policy[layer] = FloatTensor(es_models[i][pivot : pivot + sp].reshape(agent_net[layer].shape))
                pivot += sp
        policies.append(policy)
    return policies

# Nearest neighbor search plus empty space search
def search_empty_space_policies(algo, directory, start, end, env, agent_num=10):
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
    print(points)
    for p in points:
        a = empty_center(dt, p.reshape(1, -1), neigh, use_momentum=True, movestep=0.001, numiter=400)
        policies.append(a[1])
    policies = np.concatenate(policies)
    print(policies.shape)

    quit()

    agents = dump_weights(algo.agent.state_dict(), policies)

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

env_name = "Ant-v5"
env = gym.make(env_name)
# print(env.action_space, env.observation_space)

n_steps_per_rollout = 200

START_ITER = 1000
SEARCH_INTERV = 10
NUM_ITERS = START_ITER + 100

model = PPO("MlpPolicy", env, verbose=0, seed=0, n_steps=n_steps_per_rollout, batch_size=50, device='cpu', tensorboard_log='logs/'+env_name+"/")

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

exp = "PPO_empty_space"
DIR = env_name + "/" + exp + "_" + str(get_latest_run_id('logs/'+env_name+"/", exp)+1)

distanceArray = []
start_time = time.time()
timeArray = []

for i in range(START_ITER, NUM_ITERS, SEARCH_INTERV):
    print(i)
    model.learn(total_timesteps=SEARCH_INTERV*n_steps_per_rollout, 
                iteration_number_for_log = i+1,
                log_interval=1, 
                tb_log_name=exp, 
                reset_num_timesteps=True if i == START_ITER else False, 
                first_iteration=True if i == START_ITER else False,
                )
    
    agents, distance = search_empty_space_policies(model, DIR, i + 1, i + SEARCH_INTERV + 1, env)

    if i == START_ITER +10:
        break

# ret = 0
# for i in range(10):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     ret += reward
#     print(f'Reward: {reward}')

env.close()