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
            q2_net: list[nn.Module] = []
            last_layer_dim_q = self.obs_dim + self.action_dim
            # save dimensions of layers in policy and value nets
            for curr_layer_dim in self.q_layers_dims:
                q1_net.append(nn.Linear(last_layer_dim_q, curr_layer_dim))
                q2_net.append(nn.Linear(last_layer_dim_q, curr_layer_dim))
                last_layer_dim_q = curr_layer_dim
            q1_net.append(nn.Linear(last_layer_dim_q, 1))
            q2_net.append(nn.Linear(last_layer_dim_q, 1))
            self.qf1 = nn.Sequential(*q1_net).to(self.device)
            self.qf1_optim = Adam(self.qf1.parameters(), lr=self.lr)
            self.qf2 = nn.Sequential(*q2_net).to(self.device)
            self.qf2_optim = Adam(self.qf2.parameters(), lr=self.lr)
            self.qf1_target = copy.deepcopy(self.qf1)
            self.qf2_target = copy.deepcopy(self.qf2)
            
            self.model = model
            
    def train(self, model, batch_size, N_EPOCHS):
        n_updates = model.replay_buffer.buffer_size // batch_size
        for n in range(N_EPOCHS):
            for i in range(n_updates):
                batch = model.replay_buffer.sample(batch_size)
                q_loss = self.update(batch)
        return q_loss

    def update(self, rollout_data):
        obs = rollout_data.observations.to(dtype=torch.float32, device=self.device)
        actions = rollout_data.actions.to(dtype=torch.float32, device=self.device)
        rewards = rollout_data.rewards.to(dtype=torch.float32, device=self.device)
        next_obs = rollout_data.next_observations
        dones = rollout_data.dones.to(dtype=torch.float32, device=self.device)
        
        # obs_tensor = obs_as_tensor(obs, self.device)
        # next_obs_tensor = obs_as_tensor(next_obs, self.device)
        # Compute target Q-values
        with torch.no_grad():
            next_action, _, _ = self.model.policy(next_obs)
            next_action = next_action.to(dtype=torch.float32, device=self.device)
            next_obs = next_obs.to(dtype=torch.float32, device=self.device)
            target_q1 = self.qf1_target(torch.cat([next_obs, next_action], dim=-1))
            target_q2 = self.qf2_target(torch.cat([next_obs, next_action], dim=-1))
            min_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * min_q
        # Compute loss
        q1 = self.qf1(torch.cat([obs, actions], dim=-1))
        q2 = self.qf2(torch.cat([obs, actions], dim=-1))
        qf1_loss = nn.HuberLoss(delta=1.0, reduction='mean')(q1, target_q)
        qf2_loss = nn.HuberLoss(delta=1.0, reduction='mean')(q2, target_q)
        # Optimize the Q-function
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()
        self.soft_target_update()
        return (qf1_loss.item() + qf2_loss.item()) / 2.0
    
    def predict(self, obs, actions, horizon=1000):
        obs_act = torch.cat((obs, actions), dim=1)
        obs_act = obs_act.to(dtype=torch.float32, device=self.device)
        q1_pred = self.qf1(obs_act)
        q2_pred = self.qf2(obs_act)
        return torch.min(q1_pred, q2_pred).mean().item() * horizon
    
    def soft_target_update(self):
        def soft_update_from_to(source, target):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
        soft_update_from_to(self.qf1, self.qf1_target)
        soft_update_from_to(self.qf2, self.qf2_target)