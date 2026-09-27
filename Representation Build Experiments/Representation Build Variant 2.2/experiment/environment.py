import numpy as np
import cv2
import gymnasium as gym

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

class Environment(gym.Wrapper):
    def __init__(self, seed=None):
        env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8
        )

    def reset(self, **kwargs):
        true_obs, info = self.env.reset(**kwargs)
        info["situation_label"] = compute_situation(true_obs)
        info["true_state"] = true_obs
        return render_frame(self.env), info

    def step(self, action):
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        info["situation_label"] = compute_situation(true_obs)
        info["true_state"] = true_obs
        return render_frame(self.env), reward, terminated, truncated, info
