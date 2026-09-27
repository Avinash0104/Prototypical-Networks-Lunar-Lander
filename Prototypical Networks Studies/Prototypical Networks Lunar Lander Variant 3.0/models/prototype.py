import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeBank(nn.Module):
    def __init__(self, num_prototypes=32, z_dim=64, tau=0.1, num_iters=3, queue_size=128):
        super().__init__()
        self.tau = tau
        self.num_iters = num_iters
        self.queue_size = queue_size
        prototypes = torch.randn(num_prototypes, z_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = nn.Parameter(prototypes)

        queue = torch.zeros(queue_size, z_dim)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, z):
        return z @ self.prototypes.t() / self.tau

    def renormalize(self):
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

    def update_queue(self, z):
        z = F.normalize(z.detach(), dim=-1)
        scores = (z @ self.prototypes.t()).T
        p = F.softmax(scores, dim=1)
        idx = torch.distributions.Categorical(p).sample()

        num_protos = self.prototypes.shape[0]
        ptr = int(self.queue_ptr.item())
        write_idx = (ptr + torch.arange(num_protos, device=z.device)) % self.queue_size
        self.queue[write_idx] = z[idx]
        self.queue_ptr[0] = (ptr + num_protos) % self.queue_size

    def compute_reward(self, z, topk=3):
        z = F.normalize(z, dim=-1)
        distance = torch.cdist(z, self.queue, p=2)
        nearest, _ = distance.topk(topk, dim=-1, largest=False)
        return nearest[:, -1:]

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
