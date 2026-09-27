import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, z_dim=64, frame_stack=4, frame_size=84):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack, frame_size, frame_size)
            conv_dim = self.conv(dummy).shape[1]
        self.fc = nn.Linear(conv_dim, z_dim)

    def forward(self, obs):
        obs = obs.float() / 255.0
        h = self.conv(obs)
        return self.fc(h)