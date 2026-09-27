import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):

    def __init__(self, z_dim=64, action_dim=2):
        super().__init__()
        self.mean_layer = nn.Linear(z_dim, action_dim)
        self.log_std_layer = nn.Linear(z_dim, action_dim)

    def forward(self, z):
        mean = self.mean_layer(z)
        log_std = self.log_std_layer(z)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, z):
        mean, log_std = self.forward(z)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)

        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, z):
        mean, _ = self.forward(z)
        action = torch.tanh(mean)
        return action

class Critic(nn.Module):
    
    def __init__(self, z_dim=64, action_dim=2, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, action):
        h = torch.cat([z, action], dim=-1)
        return self.q1(h), self.q2(h)