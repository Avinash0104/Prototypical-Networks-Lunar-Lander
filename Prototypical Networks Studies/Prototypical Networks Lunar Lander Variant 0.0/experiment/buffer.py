import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim=136, action_dim=2, true_obs_dim=8):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.situation = np.zeros((capacity, 1), dtype=np.float32)
        self.true_obs = np.zeros((capacity, true_obs_dim), dtype=np.float32)  # Assuming the last 32 dims are noise

    def add(self, obs, action, reward, next_obs, done, situation, true_obs):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.true_obs[self.ptr] = true_obs
        self.situation[self.ptr] = situation

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[index]),
            torch.from_numpy(self.actions[index]),
            torch.from_numpy(self.rewards[index]),
            torch.from_numpy(self.next_obs[index]),
            torch.from_numpy(self.dones[index]),
            torch.from_numpy(self.situation[index]),
            torch.from_numpy(self.true_obs[index])
        )