import torch
from torch.optim import Adam
import torch.nn as nn
import copy
from stable_baselines3.common.utils import obs_as_tensor
class FQE:
    def __init__(self, obs_dim, action_dim, lr, tau=5e-3, gamma=0.98, q_layers_dims=[128, 128], device='cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.q_layers_dims = q_layers_dims
        self.device = device
        
    def build_q_net(self, model) -> None:
            q1_net: list[nn.Module] = []
            # q2_net: list[nn.Module] = []
            last_layer_dim_q = self.obs_dim + self.action_dim
            # save dimensions of layers in policy and value nets
            for curr_layer_dim in self.q_layers_dims:
                q1_net.append(nn.Linear(last_layer_dim_q, curr_layer_dim))
                # q2_net.append(nn.Linear(last_layer_dim_q, curr_layer_dim))
                last_layer_dim_q = curr_layer_dim
            q1_net.append(nn.Linear(last_layer_dim_q, 1))
            # q2_net.append(nn.Linear(last_layer_dim_q, 1))
            self.qf1 = nn.Sequential(*q1_net).to(self.device)
            self.qf1_optim = Adam(self.qf1.parameters(), lr=self.lr)
            # self.qf2 = nn.Sequential(*q2_net).to(self.device)
            # self.qf2_optim = Adam(self.qf2.parameters(), lr=lr(1))
            self.qf1_target = copy.deepcopy(self.qf1)
            # self.qf2_target = copy.deepcopy(self.qf2)
            
            self.model = model
            
    def train(self, model, batch_size, N_EPOCHS):
        n_updates = model.rollout_buffer.buffer_size // batch_size
        for _ in range(N_EPOCHS):
            for _ in range(n_updates):
                batch = model.replay_buffer.sample(batch_size)
                q_loss = self.update(batch)
                self.soft_target_update()
        return q_loss

    def update(self, rollout_data):
        obs = rollout_data.observations
        rewards = rollout_data.rewards
        next_obs = rollout_data.next_observations
        dones = rollout_data.dones
        
        # obs_tensor = obs_as_tensor(obs, self.device)
        # next_obs_tensor = obs_as_tensor(next_obs, self.device)
        # Compute target Q-values
        with torch.no_grad():
            action, _, _ = self.model.policy(obs)
            next_action, _, _ = self.model.policy(next_obs)
            obs, next_obs = obs.to(dtype=torch.float32), next_obs.to(dtype=torch.float32)
            target_q1 = self.qf1_target(torch.cat([next_obs, next_action], dim=-1))
            # target_q2 = self.qf2_target(torch.cat([next_obs, action], dim=-1))
            target_q = rewards + (1 - dones) * self.gamma * target_q1
        # Compute loss
        q1 = self.qf1(torch.cat([obs, action], dim=-1))
        # q2 = self.qf2(torch.cat([obs, action], dim=-1))
        qf1_loss = nn.MSELoss()(q1, target_q)
        # qf2_loss = nn.MSELoss()(q2, target_q)
        # Optimize the Q-function
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        # self.qf2_optim.zero_grad()
        # qf2_loss.backward()
        # self.qf2_optim.step()
        return qf1_loss.item()
    
    def soft_target_update(self):
        def soft_update_from_to(source, target):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
        soft_update_from_to(self.qf1, self.qf1_target)
        # soft_update_from_to(self.qf2, self.qf2_target, self.tau)