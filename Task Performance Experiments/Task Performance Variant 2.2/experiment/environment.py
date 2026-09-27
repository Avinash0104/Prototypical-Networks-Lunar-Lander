import numpy as np
import cv2
import gymnasium as gym
import torch

ALT_HIGH = 0.6
ALT_LOW = 0.3
SPEED_CRITICAL = 0.6
TILT_CRITICAL = 0.4

FRAME_SIZE = 84
FRAME_CHANNELS = 3  # RGB

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

def render_frame(env):
    frame = env.render()
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return frame.transpose(2, 0, 1)

def frame_diff(obs, next_obs):
    obs_last = obs[:, -FRAME_CHANNELS:].float()
    next_last = next_obs[:, -FRAME_CHANNELS:].float()
    return next_last - obs_last

class Environment(gym.Wrapper):
    def __init__(self, seed=None, encoder=None):
        env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        super().__init__(env)
        self.encoder = encoder
        self.prev_frame = None

        if self.encoder is not None:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.device = next(self.encoder.parameters()).device
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._z_dim(),), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(3, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8
            )

    def _z_dim(self):
        dummy = torch.zeros(1, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE, device=self.device)
        with torch.no_grad():
            return self.encoder(dummy).shape[1]

    def _encoder(self, frame):
        obs_t = torch.from_numpy(self.prev_frame).float().unsqueeze(0).to(self.device)
        next_obs_t = torch.from_numpy(frame).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.encoder(frame_diff(obs_t, next_obs_t))
        return z.squeeze(0).cpu().numpy()

    def reset(self, **kwargs):
        true_obs, info = self.env.reset(**kwargs)
        info["situation_label"] = compute_situation(true_obs)
        info["true_state"] = true_obs
        frame = render_frame(self.env)

        if self.encoder is not None:
            self.prev_frame = frame
            obs = self._encoder(frame)
        else:
            obs = frame

        return obs, info

    def step(self, action):
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        info["situation_label"] = compute_situation(true_obs)
        info["true_state"] = true_obs
        frame = render_frame(self.env)
        if self.encoder is not None:
            obs = self._encoder(frame)
            self.prev_frame = frame
        else:
            obs = frame
        return obs, reward, terminated, truncated, info
