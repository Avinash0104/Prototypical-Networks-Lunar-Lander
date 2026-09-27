import torch
import torch.nn.functional as F

def prototype_loss(support_z, support_y, query_z, query_y, num_classes):
    prototypes = torch.stack([support_z[support_y == c].mean(dim=0) for c in range(num_classes)])
    distances = torch.cdist(query_z, prototypes, p=2).pow(2)
    log_p_y = F.log_softmax(-distances, dim=1)
    loss = F.nll_loss(log_p_y, query_y)
    acc = (log_p_y.argmax(dim=1) == query_y).float().mean()
    return loss, acc

