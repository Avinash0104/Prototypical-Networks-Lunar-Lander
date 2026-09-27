import numpy as np
import torch

from experiment.environment import FRAME_SIZE, FRAME_STACK, FRAME_CHANNELS

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.frames = np.zeros((capacity, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.steps_since_reset = np.zeros(capacity, dtype=np.int32)
        self.dones = np.zeros(capacity, dtype=bool)

    def add(self, frame, steps_since_reset, done=False):
        self.frames[self.ptr] = frame
        self.steps_since_reset[self.ptr] = steps_since_reset
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack(self, index):
        stacks = np.zeros((len(index), FRAME_STACK, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        for k in range(FRAME_STACK):
            back = FRAME_STACK - 1 - k
            offset = np.minimum(back, self.steps_since_reset[index])
            src = (index - offset) % self.capacity
            stacks[:, k] = self.frames[src]
        return stacks.reshape(len(index), FRAME_STACK * FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE)

    def sample(self, batch_size):
        latest = (self.ptr - 1) % self.capacity
        index = np.random.randint(0, self.size, size=batch_size)
        mask = (index == latest) | self.dones[index]
        while mask.any():
            index[mask] = np.random.randint(0, self.size, size=int(mask.sum()))
            mask = (index == latest) | self.dones[index]

        next_index = (index + 1) % self.capacity
        obs = self._stack(index)
        next_obs = self._stack(next_index)

        return (
            torch.from_numpy(obs),
            torch.from_numpy(next_obs),
        )
