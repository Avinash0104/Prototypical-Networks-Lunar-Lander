import numpy as np
import gymnasium as gym

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

class Noise:
    def __init__(self, dims=32, mode="walk", walk_std=0.02, iid_std=1.0, clip=3.0, seed=None):
        assert mode in ("walk", "iid")
        self.dims = dims
        self.mode = mode
        self.walk_std = walk_std
        self.iid_std = iid_std
        self.clip = clip
        self._range = np.random.default_rng(seed)
        self._state = np.zeros(dims, dtype=np.float32)

    def reset(self):
        self._state = self._range.normal(0.0, 1.0, size=self.dims)
        self._state = np.clip(self._state, -self.clip, self.clip).astype(np.float32)
        return self._state

    def step(self):
        if self.mode == "walk":
            increment = self._range.normal(0.0, self.walk_std, size=self.dims)
            self._state += increment
        else:
            self._state = self._range.normal(0.0, self.iid_std, size=self.dims)
        self._state = np.clip(self._state, -self.clip, self.clip).astype(np.float32)
        return self._state
    

class Environment(gym.Wrapper):
    def __init__(self, noise_dims=128, noise_mode="walk", seed=None):

        env = gym.make("LunarLanderContinuous-v3")
        super().__init__(env)

        self.noise = Noise(dims=noise_dims, mode=noise_mode, seed=seed)
        noise_low = np.full(noise_dims, -self.noise.clip, dtype=np.float32)
        noise_high = np.full(noise_dims, self.noise.clip, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, noise_low]),
            high=np.concatenate([env.observation_space.high, noise_high]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        true_obs, info = self.env.reset(**kwargs)
        noise_vals = self.noise.reset()
        info["situation_label"] = compute_situation(true_obs)
        info["true_obs"] = true_obs.astype(np.float32)
        obs = np.concatenate([true_obs, noise_vals]).astype(np.float32)
        return obs, info

    def step(self, action):
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        noise_vals = self.noise.step()
        info["situation_label"] = compute_situation(true_obs)
        info["true_obs"] = true_obs.astype(np.float32)
        obs = np.concatenate([true_obs, noise_vals]).astype(np.float32)
        return obs, reward, terminated, truncated, info