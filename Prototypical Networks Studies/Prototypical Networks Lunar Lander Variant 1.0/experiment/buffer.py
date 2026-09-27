import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, frame_size=84, frame_stack=4, action_dim=2):
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.ptr = 0
        self.size = 0

        self.frames = np.zeros((capacity, frame_size, frame_size), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.situation = np.zeros((capacity, 1), dtype=np.float32)
        self.starts = np.zeros(capacity, dtype=bool) 
        self.valid = np.zeros(capacity, dtype=bool)

    def add(self, frame, action, reward, done, situation, is_start):
        self.frames[self.ptr] = frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.situation[self.ptr] = situation
        self.starts[self.ptr] = is_start
        self.valid[self.ptr] = True

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def terminate(self, frame):
        self.frames[self.ptr] = frame
        self.starts[self.ptr] = False
        self.valid[self.ptr] = False
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def stack(self, index: np.ndarray) -> np.ndarray:
        boundary = self.ptr if self.size == self.capacity else -1
        out = np.empty((index.shape[0], self.frame_stack, *self.frames.shape[1:]), dtype=np.uint8)
        for i, idx in enumerate(index):
            idx = int(idx)
            indexes = [idx]
            current = idx
            for _ in range(self.frame_stack - 1):
                if self.starts[current] or current == boundary:
                    break
                current = (current - 1) % self.capacity
                indexes.append(current)
            indexes = indexes[::-1]
            while len(indexes) < self.frame_stack:
                indexes.insert(0, indexes[0])
            out[i] = self.frames[indexes]
        return out

    def sample(self, batch_size):
        last_index = (self.ptr - 1) % self.capacity
        index = np.random.randint(0, self.size, size=batch_size)
        invalid = (index == last_index) | (~self.valid[index])
        while invalid.any():
            index[invalid] = np.random.randint(0, self.size, size=invalid.sum())
            invalid = (index == last_index) | (~self.valid[index])

        next_index = (index + 1) % self.capacity
        obs = self.stack(index)
        next_obs = self.stack(next_index)

        return (
            torch.from_numpy(obs),
            torch.from_numpy(self.actions[index]),
            torch.from_numpy(self.rewards[index]),
            torch.from_numpy(next_obs),
            torch.from_numpy(self.dones[index]),
            torch.from_numpy(self.situation[index]),
        )