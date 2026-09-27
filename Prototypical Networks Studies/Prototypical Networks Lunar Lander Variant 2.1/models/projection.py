import torch.nn as nn

class ProjectionHead(nn.Module):
    
    def __init__(self, z_dim=64, u_dim=32):
        super().__init__()
        self.net = nn.Linear(z_dim, u_dim)

    def forward(self, z):
        return self.net(z)