import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, frame_stack=4, frame_size=84, action_dim=2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.frame_stack = frame_stack
        self.frame_size = frame_size

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

    def terminate(self, frame, situation):
        self.frames[self.ptr] = frame
        self.situation[self.ptr] = situation
        self.starts[self.ptr] = False
        self.valid[self.ptr] = False

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def stack_indices(self, index, boundary):
        indexes = [index]
        current = index
        for _ in range(self.frame_stack - 1):
            if self.starts[current] or current == boundary:
                break
            current = (current - 1) % self.capacity
            indexes.append(current)
        indexes = indexes[::-1]
        while len(indexes) < self.frame_stack:
            indexes.insert(0, indexes[0])
        return indexes

    def get_stacks(self, indices):
        boundary = self.ptr if self.size == self.capacity else -1
        stacks = np.empty((len(indices), self.frame_stack, self.frame_size, self.frame_size), dtype=np.uint8)
        for i, index in enumerate(indices):
            stacks[i] = self.frames[self.stack_indices(int(index), boundary)]
        return stacks

    def sample_transition_indices(self, batch_size):
        last_index = (self.ptr - 1) % self.capacity
        index = np.random.randint(0, self.size, size=batch_size)
        invalid = (index == last_index) | (~self.valid[index])
        while invalid.any():
            index[invalid] = np.random.randint(0, self.size, size=int(invalid.sum()))
            invalid = (index == last_index) | (~self.valid[index])
        return index

    def sample(self, batch_size):
        index = self.sample_transition_indices(batch_size)
        next_index = (index + 1) % self.capacity
        obs = self.get_stacks(index)
        next_obs = self.get_stacks(next_index)
        return (
            torch.from_numpy(obs),
            torch.from_numpy(self.actions[index]),
            torch.from_numpy(self.rewards[index]),
            torch.from_numpy(next_obs),
            torch.from_numpy(self.dones[index]),
            torch.from_numpy(self.situation[index]),
        )

    def sample_episode(self, n_support, n_query, num_classes=4):
        valid_mask = self.valid[:self.size]
        labels = self.situation[:self.size, 0]
        n = n_support + n_query
        support_index, query_index = [], []
        for c in range(num_classes):
            class_index = np.where((labels == c) & valid_mask)[0]
            if len(class_index) == 0: return None
            chosen = np.random.choice(class_index, size=n, replace=len(class_index) < n)
            support_index.append(chosen[:n_support])
            query_index.append(chosen[n_support:])
        support_index = np.concatenate(support_index)
        query_index = np.concatenate(query_index)
        support_obs = self.get_stacks(support_index)
        query_obs = self.get_stacks(query_index)
        return (
            torch.from_numpy(support_obs), torch.from_numpy(self.situation[support_index]).long().squeeze(-1),
            torch.from_numpy(query_obs), torch.from_numpy(self.situation[query_index]).long().squeeze(-1),
        )
