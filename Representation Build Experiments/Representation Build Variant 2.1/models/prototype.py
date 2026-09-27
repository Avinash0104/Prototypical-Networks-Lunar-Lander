import torch
import torch.nn.functional as F

def prototypical_loss(support_z: torch.Tensor, support_y: torch.Tensor,
                       query_z: torch.Tensor, query_y: torch.Tensor, num_classes: int):
    prototypes = torch.stack([support_z[support_y == c].mean(dim=0) for c in range(num_classes)])

    dists = torch.cdist(query_z, prototypes, p=2).pow(2)
    log_p = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p, query_y)

    with torch.no_grad():
        accuracy = (log_p.argmax(dim=1) == query_y).float().mean()

    return loss, accuracy

