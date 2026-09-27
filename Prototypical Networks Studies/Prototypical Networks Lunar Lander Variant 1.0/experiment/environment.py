import cv2
import numpy as np
import gymnasium as gym
from collections import deque

ALT_HIGH = 0.6
ALT_LOW = 0.3
SPEED_CRITICAL = 0.6
TILT_CRITICAL = 0.4

def compute_situation(state: np.ndarray) -> int:
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state
    speed = np.hypot(vx, vy)
    tilt = abs(angle)

    if left_leg > 0.5 or right_leg > 0.5:
        return 3
    if y < ALT_LOW and (speed > SPEED_CRITICAL or tilt > TILT_CRITICAL):
        return 2
    if y > ALT_HIGH:
        return 0
    return 1

class FrameStacker:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame: np.ndarray):
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.stack()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return self.stack()

    def stack(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)

class Environment(gym.Wrapper):
    def __init__(self, frame_size=84, seed=None):
        env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        super().__init__(env)
        self.frame_size = frame_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(frame_size, frame_size), dtype=np.uint8)

    def _frame(self) -> np.ndarray:
        raw = self.env.render()
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        frame = resized.astype(np.uint8)
        return frame

    def reset(self, **kwargs):
        true_state, info = self.env.reset(**kwargs)
        info["situation_label"] = compute_situation(true_state)
        obs = self._frame()
        return obs, info

    def step(self, action):
        true_state, reward, terminated, truncated, info = self.env.step(action)
        info["situation_label"] = compute_situation(true_state)
        obs = self._frame()
        return obs, reward, terminated, truncated, info