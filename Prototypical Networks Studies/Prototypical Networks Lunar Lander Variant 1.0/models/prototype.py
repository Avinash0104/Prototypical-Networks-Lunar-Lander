import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeBank(nn.Module):
    def __init__(self, num_prototypes=4, u_dim=32, tau=0.1):
        super().__init__()
        self.tau = tau
        prototypes = torch.randn(num_prototypes, u_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = nn.Parameter(prototypes)

    def forward(self, u):
        return u @ self.prototypes.t() / self.tau

    def renormalize(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)