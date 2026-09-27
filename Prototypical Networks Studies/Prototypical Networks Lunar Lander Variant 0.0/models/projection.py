import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    
    def __init__(self, z_dim=64, u_dim=32):
        super().__init__()
        self.net = nn.Linear(z_dim, u_dim)

    def forward(self, z):
        u = self.net(z)
        return F.normalize(u, dim=-1)