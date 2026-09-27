import numpy as np
import torch

from experiment.environment import FRAME_SIZE, FRAME_CHANNELS
from experiment.clustering import assign_clusters, diff_feature, frame_diff

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.frames = np.zeros((capacity, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.steps_since_reset = np.zeros(capacity, dtype=np.int32)

    def add(self, frame, steps_since_reset):
        self.frames[self.ptr] = frame
        self.steps_since_reset[self.ptr] = steps_since_reset

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _random_valid_indices(self, n):
        latest = (self.ptr - 1) % self.capacity
        index = np.random.randint(0, self.size, size=n)
        mask = (index == latest) | (self.steps_since_reset[(index + 1) % self.capacity] == 0)
        while mask.any():
            index[mask] = np.random.randint(0, self.size, size=int(mask.sum()))
            mask = (index == latest) | (self.steps_since_reset[(index + 1) % self.capacity] == 0)
        return index

    def _last_frame(self, index):
        return self.frames[index]

    def gather_last(self, index):
        next_index = (index + 1) % self.capacity
        obs_last = self._last_frame(index)
        next_obs_last = self._last_frame(next_index)
        return obs_last, next_obs_last

    def sample_by_cluster(self, centroids, n_classes, num_support, num_query, candidate_pool=4000, pool_size=8):
        needed_per_class = num_support + num_query
        pool = min(candidate_pool, self.size)
        index = self._random_valid_indices(pool)

        obs_last, next_obs_last = self.gather_last(index)

        feats = diff_feature(torch.from_numpy(obs_last), torch.from_numpy(next_obs_last), pool_size=pool_size).numpy()
        assignments = assign_clusters(feats, centroids)

        groups = {}
        for pos, cluster_id in enumerate(assignments):
            groups.setdefault(int(cluster_id), []).append(pos)

        eligible = [c for c, members in groups.items() if len(members) >= needed_per_class]
        if not eligible:
            return None

        rng = np.random.default_rng()
        chosen = rng.choice(eligible, size=min(n_classes, len(eligible)), replace=False)

        support_pos_list, query_pos_list = [], []
        for c in chosen:
            members = rng.permutation(groups[int(c)])
            support_pos_list.append(members[:num_support])
            query_pos_list.append(members[num_support:num_support + num_query])

        # obs_last/next_obs_last are already position-aligned over the whole candidate
        # pool, so the chosen support/query positions index directly into them -- no
        # extra buffer lookup or frame-stacking needed for the handful actually used.
        support_pos = np.concatenate(support_pos_list)
        query_pos = np.concatenate(query_pos_list)

        C = len(chosen)
        support_diff = frame_diff(torch.from_numpy(obs_last[support_pos]), torch.from_numpy(next_obs_last[support_pos]))
        query_diff = frame_diff(torch.from_numpy(obs_last[query_pos]), torch.from_numpy(next_obs_last[query_pos]))

        support_next_obs = support_diff.reshape(C, num_support, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE)
        query_next_obs = query_diff.reshape(C, num_query, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE)
        return support_next_obs, query_next_obs
