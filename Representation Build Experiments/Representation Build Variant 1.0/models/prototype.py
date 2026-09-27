import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeBank(nn.Module):
    def __init__(self, num_prototypes=32, z_dim=64, tau=0.1, num_iters=3):
        super().__init__()
        self.tau = tau
        self.num_iters = num_iters
        prototypes = torch.randn(num_prototypes, z_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = nn.Parameter(prototypes)

    def forward(self, z):
        return z @ self.prototypes.t() / self.tau

    def renormalize(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

    def sinkhorn(self, scores):
        def remove_infs(x):
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores - scores.max()
        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for _ in range(self.num_iters):
            u = Q.sum(dim=1)
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T
