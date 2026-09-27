import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=4, z_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.fc = nn.Linear(conv_out, z_dim)

    def forward(self, obs):
        obs = obs.float() / 255.0
        h = self.conv(obs)
        return self.fc(h)