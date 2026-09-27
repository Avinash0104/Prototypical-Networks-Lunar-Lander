import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from models.encoder import Encoder
from models.sac import Actor, Critic
from experiment.environment import Environment
from experiment.buffer import ReplayBuffer, frame_diff
from experiment.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint, load_pretrained_encoder

PIXEL_BUFFER_CAPACITY = 50_000

class Agent:
    def __init__(self, z_dim, action_dim=2, gamma=0.99, tau=0.005, lr=3e-4, device="cpu", encoder=None):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.encoder = (encoder if encoder is not None else nn.Identity()).to(device)

        self.actor = Actor(z_dim, action_dim).to(device)
        self.critic = Critic(z_dim, action_dim).to(device)

        self.target = copy.deepcopy(self.critic).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        # encoder params ride along in the critic optimizer -- only the critic loss
        # is allowed to shape the representation (see update_actor/update_alpha below)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()), lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def soft_update(self):
        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_critic(self, obs, action, reward, next_obs, done):
        z = self.encoder(obs)
        with torch.no_grad():
            next_z = self.encoder(next_obs)
            next_action, next_log_prob = self.actor.sample(next_z)
            target_q1, target_q2 = self.target(next_z, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = reward + self.gamma * (1 - done) * target_q

        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        return critic_loss, z.detach()

    def update_actor(self, z):
        action, log_prob = self.actor.sample(z)
        q1, q2 = self.critic(z, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        return actor_loss

    def update_alpha(self, z):
        action, log_prob = self.actor.sample(z)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10.0, 5.0)  # keeps alpha in [~4.5e-5, ~148]; unbounded, it can explode over many steps

        return alpha_loss


    def update(self, batch):
        obs, next_obs, action, reward, done = batch

        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        critic_loss, z = self.update_critic(obs, action, reward, next_obs, done)
        actor_loss = self.update_actor(z)
        alpha_loss = self.update_alpha(z)
        self.soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def select_action(self, obs, deterministic=False):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.encoder(obs_t)
            if deterministic:
                action = self.actor.deterministic(z)
            else:
                action, _ = self.actor.sample(z)
        return action.squeeze(0).cpu().numpy()
    

def train(num_steps=300_000, warmup_steps=1000, batch_size=256, buffer_capacity=200_000, seed=0, arm="trained", pretrained_step=100_000, logstep=10_000, result="results", resume=True, device="cpu"):
    run_name = f"{arm}_seed{seed}"
    dir = os.path.join(result, run_name)
    os.makedirs(dir, exist_ok=True)

    torch.manual_seed(seed)
    if arm == "true_state":
        env = gym.make("LunarLanderContinuous-v3")
        obs_shape = env.observation_space.shape
        pixel_arm = False
        agent = Agent(z_dim=obs_shape[0], device=device)
        buffer = ReplayBuffer(capacity=buffer_capacity, obs_shape=obs_shape, dtype=np.float32)
    elif arm in ("trained", "random_floor"):
        env = Environment(seed=seed)
        obs_shape = env.observation_space.shape  # (3, 84, 84) raw frame
        pixel_arm = True
        encoder = Encoder().to(device)
        if arm == "trained":
            pretrained_path = os.path.join("checkpoints", f"prototype_seed{seed}", f"checkpoint_{pretrained_step}.pt")
            load_pretrained_encoder(encoder, pretrained_path, device=device)
        agent = Agent(z_dim=64, device=device, encoder=encoder)
        buffer = ReplayBuffer(capacity=min(buffer_capacity, PIXEL_BUFFER_CAPACITY), obs_shape=obs_shape, dtype=np.int16)
    else:
        raise ValueError(f"unknown condition: {arm}")

    start_step = 0
    episode_returns = []
    episode_steps = []

    if resume:
        checkpoint_path, checkpoint_step = find_latest_checkpoint(dir)
        if checkpoint_path is not None:
            load_checkpoint(agent, checkpoint_path)
            start_step = checkpoint_step
            returns_path = os.path.join(dir, "episode_returns.npy")
            if os.path.exists(returns_path):
                episode_returns = list(np.load(returns_path))
            steps_path = os.path.join(dir, "episode_steps.npy")
            if os.path.exists(steps_path):
                episode_steps = list(np.load(steps_path))
            print(f"[{run_name}] resuming from step {start_step} ({len(episode_returns)} episodes so far)", flush=True)

    frame, info = env.reset(seed=seed)
    if pixel_arm:
        prev_frame = frame
        obs = frame_diff(prev_frame, frame)  # zero at the very start, same as before
    else:
        obs = frame
    episode_return = 0.0

    for step in range(start_step, num_steps):
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_frame, reward, terminated, truncated, info = env.step(action)
        done = float(terminated)
        buffer.add(obs, action, reward, done)

        episode_return += reward

        if pixel_arm:
            prev_frame = frame
            frame = next_frame
            obs = frame_diff(prev_frame, frame)
        else:
            obs = next_frame

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_steps.append(step + 1)
            if len(episode_returns) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                print(f"step={step}  episode={len(episode_returns)} avg_return(last10)={avg_return:.1f}")
            episode_return = 0.0
            frame, info = env.reset()
            if pixel_arm:
                prev_frame = frame
                obs = frame_diff(prev_frame, frame)
            else:
                obs = frame

        if step >= warmup_steps and buffer.size >= batch_size:
            agent.update(buffer.sample(batch_size))

        if (step + 1) % logstep == 0:
            save_checkpoint(agent, os.path.join(dir, f"checkpoint_{step + 1}.pt"))
            np.save(os.path.join(dir, "episode_returns.npy"), np.array(episode_returns))
            np.save(os.path.join(dir, "episode_steps.npy"), np.array(episode_steps))

    return episode_returns


if __name__ == "__main__":
    train()





    