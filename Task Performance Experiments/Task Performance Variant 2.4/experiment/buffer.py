import numpy as np
import torch


def frame_diff(obs, next_obs):
    return next_obs.astype(np.float32) - obs.astype(np.float32)


class ReplayBuffer:
    """
    Stores one observation per env step (plus the action/reward/done taken from it),
    and reconstructs each transition's next_obs from the following slot at sample
    time -- so nothing is ever stored twice. Episode boundaries don't need special
    handling: a transition that ends an episode (done=1) has its next_obs zeroed
    out by the (1 - done) term in the Bellman target anyway, so it doesn't matter
    if that slot happens to hold the start of a different episode.
    """

    def __init__(self, capacity, obs_shape, action_dim=2, dtype=np.float32):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # exclude the slot about to be overwritten next -- its neighbor hasn't been written yet
        latest = (self.ptr - 1) % self.capacity
        valid = np.arange(self.size)
        valid = valid[valid != latest]
        index = np.random.choice(valid, size=batch_size, replace=True)
        next_index = (index + 1) % self.capacity

        return (
            torch.from_numpy(self.obs[index]).float(),
            torch.from_numpy(self.obs[next_index]).float(),
            torch.from_numpy(self.actions[index]),
            torch.from_numpy(self.rewards[index]),
            torch.from_numpy(self.dones[index]),
        )
