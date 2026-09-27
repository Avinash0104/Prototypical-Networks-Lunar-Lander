import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from experiment.environment import FRAME_CHANNELS

def frame_diff(obs, next_obs):
    obs_last = obs[:, -FRAME_CHANNELS:].float()
    next_last = next_obs[:, -FRAME_CHANNELS:].float()
    return next_last - obs_last

def diff_feature(obs, next_obs, pool_size=8):
    diff = frame_diff(obs, next_obs).abs()
    pooled = F.adaptive_avg_pool2d(diff, (pool_size, pool_size))
    return pooled.flatten(start_dim=1)

def fit_centroids(buffer, sample_size=8000, k=16, seed=None, pool_size=8):
    sample_size = min(sample_size, buffer.size)
    index = buffer._random_valid_indices(sample_size)
    obs, next_obs = buffer.gather_last(index)

    feats = diff_feature(torch.from_numpy(obs), torch.from_numpy(next_obs), pool_size=pool_size).numpy()

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(feats)
    return kmeans.cluster_centers_

def assign_clusters(feats, centroids):
    dists = ((feats[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return dists.argmin(axis=1)
