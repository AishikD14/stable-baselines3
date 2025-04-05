import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv as AntEnvV5

class AntDirEnv(AntEnvV5):
    def __init__(self, goal=0.0, **kwargs):
        super().__init__(**kwargs)
        self.goal = goal # rad
        self.direction = (np.cos(self.goal), np.sin(self.goal))

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        forward_reward = np.dot(xy_velocity, self.direction)
        offset = np.dot(xy_velocity / np.linalg.norm(xy_velocity), self.direction)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(forward_reward, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": xy_velocity[0],
            "y_velocity": xy_velocity[1],
            "goal_offset": offset,
            **reward_info,
        }
        

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_goal(self, goal):
        self.goal = goal # rad
        self.direction = (np.cos(self.goal), np.sin(self.goal))