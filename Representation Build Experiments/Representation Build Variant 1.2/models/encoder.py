import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, frame_stack, frame_channels, z_dim=64):
        super().__init__()
        self.frame_stack = frame_stack
        self.frame_channels = frame_channels

        conv_in_channels = frame_channels * (frame_stack - 1)
        self.convnet = nn.Sequential(
            nn.Conv2d(conv_in_channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, conv_in_channels, 84, 84)
            flat_dim = self.convnet(dummy).view(1, -1).shape[1]

        self.trunk = nn.Linear(flat_dim, z_dim)

    def forward(self, obs):
        obs = obs / 255.0
        b, _, height, width = obs.shape
        frames = obs.view(b, self.frame_stack, self.frame_channels, height, width)
        diffs = frames[:, 1:] - frames[:, :-1]
        diffs = diffs.reshape(b, (self.frame_stack - 1) * self.frame_channels, height, width)

        h = self.convnet(diffs)
        h = h.view(h.shape[0], -1)
        return self.trunk(h)
