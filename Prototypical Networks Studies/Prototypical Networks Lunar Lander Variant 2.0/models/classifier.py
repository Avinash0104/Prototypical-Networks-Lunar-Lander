import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    
    def __init__(self, z_dim=64, hidden_dim=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):
        return self.net(z)