import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, obs_dim=136, hidden_dim=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, obs):
        return self.net(obs)