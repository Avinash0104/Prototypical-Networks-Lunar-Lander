import numpy as np
import torch

from experiment.environment import FRAME_SIZE, FRAME_STACK, FRAME_CHANNELS
from experiment.clustering import assign_clusters, frame_feature, POOL_SIZE

STACK_CHANNELS = FRAME_STACK * FRAME_CHANNELS

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

    def sample_indices(self, n):
        candidates = np.arange(self.size)
        if self.size == self.capacity:
            # once full, the oldest slots' look-back wraps onto the newest frames
            oldest = (self.ptr + np.arange(FRAME_STACK - 1)) % self.capacity
            candidates = np.delete(candidates, oldest)
        n = min(n, len(candidates))
        return np.random.choice(candidates, size=n, replace=False)

    def gather(self, index):
        stacks = np.zeros((len(index), FRAME_STACK, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        for k in range(FRAME_STACK):
            back = FRAME_STACK - 1 - k
            # clamp the look-back at the episode start so a stack never crosses a reset
            offset = np.minimum(back, self.steps_since_reset[index])
            src = (index - offset) % self.capacity
            stacks[:, k] = self.frames[src]
        return stacks.reshape(len(index), STACK_CHANNELS, FRAME_SIZE, FRAME_SIZE)

    def sample_by_cluster(self, centroids, n_classes, num_support, num_query, candidate_pool=4000, pool_size=POOL_SIZE):
        needed_per_class = num_support + num_query
        index = self.sample_indices(candidate_pool)

        stacks = self.gather(index)

        feats = frame_feature(torch.from_numpy(stacks), pool_size=pool_size).numpy()
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

        # stacks is already position-aligned over the whole candidate pool, so the
        # chosen support/query positions index directly into it
        support_pos = np.concatenate(support_pos_list)
        query_pos = np.concatenate(query_pos_list)

        C = len(chosen)
        support_stacks = torch.from_numpy(stacks[support_pos]).float()
        query_stacks = torch.from_numpy(stacks[query_pos]).float()

        support_obs = support_stacks.reshape(C, num_support, STACK_CHANNELS, FRAME_SIZE, FRAME_SIZE)
        query_obs = query_stacks.reshape(C, num_query, STACK_CHANNELS, FRAME_SIZE, FRAME_SIZE)
        return support_obs, query_obs
