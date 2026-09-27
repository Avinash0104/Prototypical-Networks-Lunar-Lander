import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, z_dim=64):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.convnet(dummy).reshape(1, -1).shape[1]

        self.trunk  = nn.Linear(flat_dim, z_dim)

    def forward(self, obs):
        obs = obs / 255.0
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        return self.trunk(h)
