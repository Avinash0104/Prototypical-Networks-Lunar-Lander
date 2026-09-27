import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

POOL_SIZE = 16

def frame_feature(frames, pool_size=POOL_SIZE):
    pooled = F.adaptive_avg_pool2d(frames.float(), (pool_size, pool_size))
    return pooled.flatten(start_dim=1)

def fit_centroids(buffer, sample_size=8000, k=16, seed=None, pool_size=POOL_SIZE):
    index = buffer.sample_indices(sample_size)
    frames = buffer.gather(index)

    feats = frame_feature(torch.from_numpy(frames), pool_size=pool_size).numpy()

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(feats)
    return kmeans.cluster_centers_

def assign_clusters(feats, centroids):
    # argmin ||x - c||^2 = argmin (||c||^2 - 2 x.c), since ||x||^2 is constant per row;
    # avoids materialising a (n, k, feature_dim) difference tensor on every call
    feats = feats.astype(np.float64, copy=False)
    centroids = centroids.astype(np.float64, copy=False)
    scores = (centroids ** 2).sum(axis=1)[None, :] - 2.0 * (feats @ centroids.T)
    return scores.argmin(axis=1)
