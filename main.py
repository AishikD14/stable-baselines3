import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="human")
print(env.action_space, env.observation_space)

model = PPO("MlpPolicy", env, verbose=1, seed=0, n_steps=200, batch_size=50, device='cpu')
print(model.policy)
quit()
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()