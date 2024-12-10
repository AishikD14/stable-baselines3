import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Ant-v5")
print(env.action_space, env.observation_space)

n_steps = 200
n_iterations = 1000

model = PPO("MlpPolicy", env, verbose=1, seed=0, n_steps=n_steps, batch_size=50, device='cpu', tensorboard_log='logs/')

# print("Starting Initial training")
# model.learn(total_timesteps=n_iterations*n_steps, log_interval=50)
# model.save("full_exp_on_ppo/models/ppo_ant")
# print("Initial training done")
# quit()

print("Loading model")

# Load model
model.set_parameters("full_exp_on_ppo/models/ppo_ant", device='cpu')

print("Model loaded")

vec_env = model.get_env()
obs = vec_env.reset()

print("Starting evaluation")
ret = 0
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    ret += reward
    print(f'Reward: {reward}')

env.close()