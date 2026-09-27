import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, z_dim=64, pred_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, pred_dim), nn.ReLU(),
            nn.Linear(pred_dim, z_dim)
        )

    def forward(self, z):
        return self.net(z)
