import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim=2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.obs[index]),
            torch.from_numpy(self.next_obs[index]),
            torch.from_numpy(self.actions[index]),
            torch.from_numpy(self.rewards[index]),
            torch.from_numpy(self.dones[index]),
        )