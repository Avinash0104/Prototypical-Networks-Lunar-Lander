import os
import copy
import torch
import torch.nn.functional as F
import numpy as np

from models.encoder import Encoder
from experiment.environment import Environment, FRAME_STACK, FRAME_CHANNELS
from experiment.buffer import ReplayBuffer
from models.predictor import Predictor
from models.prototype import PrototypeBank
from experiment.checkpoint import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    save_replay_buffer, load_replay_buffer,
)

class Agent:
    def __init__(self, z_dim=64, lr=1e-4, device="cpu", num_prototypes=32, pred_dim=128,
                 prototype_assignment_tau=0.1, encoder_target_tau=0.05, encoder_update_frequency=2):
        self.device = device
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency

        in_channels = FRAME_STACK * FRAME_CHANNELS
        self.encoder = Encoder(in_channels, z_dim).to(device)

        self.encoder_target = copy.deepcopy(self.encoder).to(device)
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        self.predictor = Predictor(z_dim, pred_dim).to(device)
        self.prototype_bank = PrototypeBank(num_prototypes, z_dim, prototype_assignment_tau).to(device)
        self.proto_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()) + list(self.prototype_bank.parameters()), lr=lr
        )

    def soft_update_target_encoder(self):
        for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            target_param.data.copy_(self.encoder_target_tau * param.data + (1 - self.encoder_target_tau) * target_param.data)

    def random_shift(self, obs, pad=4):
        b, c, h, w = obs.shape
        padded = F.pad(obs, (pad, pad, pad, pad), mode="replicate")

        eps = 1.0 / (h + 2 * pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=obs.device, dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)

        shift = torch.randint(0, 2 * pad + 1, size=(b, 1, 1, 2), device=obs.device)
        shift = shift.to(dtype=obs.dtype) * (2.0 / (h + 2 * pad))

        grid = base_grid + shift
        return F.grid_sample(padded, grid, padding_mode="zeros", align_corners=False)

    def update_proto_rep(self, obs_aug, next_obs_aug):
        z = self.encoder(obs_aug)
        with torch.no_grad():
            next_z = self.encoder_target(next_obs_aug)

        z_pred = F.normalize(self.predictor(z), dim=-1)
        next_z_norm = F.normalize(next_z, dim=-1)

        logits_s = self.prototype_bank(z_pred)
        log_p_s = F.log_softmax(logits_s, dim=-1)

        with torch.no_grad():
            logits_t = self.prototype_bank(next_z_norm)
            q_t = self.prototype_bank.sinkhorn(logits_t)

        proto_loss = -(q_t * log_p_s).sum(dim=-1).mean()

        self.proto_optimizer.zero_grad()
        proto_loss.backward()
        self.proto_optimizer.step()
        self.prototype_bank.renormalize()

        return proto_loss

    def update(self, batch, step):
        obs, next_obs = batch
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)

        obs = self.random_shift(obs.float())
        next_obs = self.random_shift(next_obs.float())

        loss = None
        if step % self.encoder_update_frequency == 0:
            loss = self.update_proto_rep(obs, next_obs)
            self.soft_update_target_encoder()

        return loss.item() if loss is not None else None
    

def train(num_steps=100_000, warmup=1000, batch_size=256, buffer_capacity=70_000, seed=0, logstep=10_000,
          result="results", resume=True, device="cpu"):
    run_name = f"prototype_seed{seed}"
    dir = os.path.join(result, run_name)
    os.makedirs(dir, exist_ok=True)

    env = Environment(seed=seed)
    env.action_space.seed(seed)
    agent = Agent(device=device)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    start_step = 0
    episode_returns = []

    if resume:
        checkpoint_path, checkpoint_step = find_latest_checkpoint(dir)
        if checkpoint_path is not None:
            load_checkpoint(agent, checkpoint_path)
            start_step = checkpoint_step
            returns_path = os.path.join(dir, "episode_returns.npy")
            if os.path.exists(returns_path):
                episode_returns = list(np.load(returns_path))
            buffer_restored = load_replay_buffer(buffer, dir)
            print(f"[{run_name}] resuming from step {start_step} ({len(episode_returns)} episodes so far, "
                  f"replay buffer {'restored' if buffer_restored else 'NOT found, starting empty'})", flush=True)

    frame, info = env.reset(seed=seed)
    steps_since_reset = 0
    episode_return = 0.0
    recent_losses = []

    for step in range(start_step, num_steps):
        action = env.action_space.sample()
        next_frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(frame, steps_since_reset, False)

        frame = next_frame
        steps_since_reset += 1
        episode_return += reward

        if done:
            # store the terminal frame itself (marked done) instead of discarding it;
            # `done` keeps it from being sampled as a source whose "next" slot would
            # otherwise be the following episode's unrelated reset frame
            buffer.add(frame, steps_since_reset, True)

            episode_returns.append(episode_return)
            if len(episode_returns) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                print(f"step={step}  episode={len(episode_returns)} avg_return(last10)={avg_return:.1f}")
            episode_return = 0.0
            frame, info = env.reset()
            steps_since_reset = 0

        if step >= warmup and buffer.size >= batch_size:
            loss = agent.update(buffer.sample(batch_size), step)
            if loss is not None:
                recent_losses.append(loss)

        if (step + 1) % 1000 == 0 and recent_losses:
            print(f"step={step + 1}  proto_loss(last1000avg)={np.mean(recent_losses):.4f}", flush=True)
            recent_losses = []

        if (step + 1) % logstep == 0:
            save_checkpoint(agent, os.path.join(dir, f"checkpoint_{step + 1}.pt"))
            np.save(os.path.join(dir, "episode_returns.npy"), np.array(episode_returns))
            save_replay_buffer(buffer, dir)

    return episode_returns


if __name__ == "__main__":
    train()
