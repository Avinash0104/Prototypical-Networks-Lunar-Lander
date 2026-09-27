import os
import copy
import torch
import numpy as np

from models.encoder import Encoder
from experiment.environment import Environment, FRAME_STACK, FRAME_CHANNELS
from experiment.buffer import ReplayBuffer
from experiment.clustering import fit_centroids
from models.prototype import prototypical_loss
from experiment.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

class Agent:
    def __init__(self, z_dim=64, lr=1e-4, device="cpu",
                 encoder_target_tau=0.05, encoder_update_frequency=1):
        self.device = device
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency

        in_channels = FRAME_STACK * FRAME_CHANNELS
        self.encoder = Encoder(in_channels, z_dim).to(device)

        self.encoder_target = copy.deepcopy(self.encoder).to(device)
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        self.proto_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def soft_update_target_encoder(self):
        for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            target_param.data.copy_(self.encoder_target_tau * param.data + (1 - self.encoder_target_tau) * target_param.data)

    def update_proto_motion(self, support_next_obs, query_next_obs):
        C, K = support_next_obs.shape[:2]
        Q = query_next_obs.shape[1]

        support_flat = support_next_obs.view(-1, *support_next_obs.shape[2:])
        z_support = self.encoder(support_flat)
        support_y = torch.arange(C, device=self.device).repeat_interleave(K)

        query_flat = query_next_obs.view(-1, *query_next_obs.shape[2:])
        with torch.no_grad():
            z_query = self.encoder_target(query_flat)
        query_y = torch.arange(C, device=self.device).repeat_interleave(Q)

        loss, accuracy = prototypical_loss(z_support, support_y, z_query, query_y, num_classes=C)

        self.proto_optimizer.zero_grad()
        loss.backward()
        self.proto_optimizer.step()

        return loss, accuracy

    def update(self, batch, step):
        support_next_obs, query_next_obs = batch
        support_next_obs = support_next_obs.to(self.device).float()
        query_next_obs = query_next_obs.to(self.device).float()

        loss, accuracy = None, None
        if step % self.encoder_update_frequency == 0:
            loss, accuracy = self.update_proto_motion(support_next_obs, query_next_obs)
            self.soft_update_target_encoder()

        return (loss.item(), accuracy.item()) if loss is not None else (None, None)


def train(num_steps=100_000, warmup=1000, buffer_capacity=70_000, seed=0, logstep=50_000,
          result="results", resume=True, num_classes=16, pool_size=8, num_support=4, num_query=4,
          recluster_freq=5000, centroid_sample_size=4000, candidate_pool=1000,
          device=None, print_every=1000):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = f"prototype_seed{seed}"
    dir = os.path.join(result, run_name)
    os.makedirs(dir, exist_ok=True)

    env = Environment(seed=seed)
    agent = Agent(device=device)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    start_step = 0
    episode_returns = []
    proto_losses = []
    proto_accuracies = []

    if resume:
        checkpoint_path, checkpoint_step = find_latest_checkpoint(dir)
        if checkpoint_path is not None:
            load_checkpoint(agent, checkpoint_path)
            start_step = checkpoint_step
            returns_path = os.path.join(dir, "episode_returns.npy")
            if os.path.exists(returns_path):
                episode_returns = list(np.load(returns_path))
            losses_path = os.path.join(dir, "proto_losses.npy")
            if os.path.exists(losses_path):
                proto_losses = list(np.load(losses_path))
            acc_path = os.path.join(dir, "proto_accuracies.npy")
            if os.path.exists(acc_path):
                proto_accuracies = list(np.load(acc_path))
            print(f"[{run_name}] resuming from step {start_step} ({len(episode_returns)} episodes so far)", flush=True)

    print(f"[{run_name}] training on device={device}", flush=True)

    frame, info = env.reset(seed=seed)
    steps_since_reset = 0
    episode_return = 0.0
    centroids = None
    recent_losses, recent_accuracies = [], []

    for step in range(start_step, num_steps):
        action = env.action_space.sample()
        next_frame, reward, terminated, truncated, info = env.step(action)
        buffer.add(frame, steps_since_reset)

        frame = next_frame
        steps_since_reset = 0 if (terminated or truncated) else steps_since_reset + 1
        episode_return += reward

        if terminated or truncated:
            episode_returns.append(episode_return)
            if len(episode_returns) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                print(f"step={step}  episode={len(episode_returns)} avg_return(last10)={avg_return:.1f}")
            episode_return = 0.0
            frame, info = env.reset()
            steps_since_reset = 0

        if step >= warmup and buffer.size >= centroid_sample_size and (centroids is None or step % recluster_freq == 0):
            centroids = fit_centroids(buffer, sample_size=centroid_sample_size, k=num_classes, seed=seed, pool_size=pool_size)

        if step >= warmup and centroids is not None:
            batch = buffer.sample_by_cluster(centroids, n_classes=num_classes, num_support=num_support,
                                              num_query=num_query, candidate_pool=candidate_pool, pool_size=pool_size)
            if batch is not None:
                loss, accuracy = agent.update(batch, step)
                if loss is not None:
                    recent_losses.append(loss)
                    recent_accuracies.append(accuracy)

        if (step + 1) % print_every == 0 and recent_losses:
            proto_losses.append(np.mean(recent_losses))
            proto_accuracies.append(np.mean(recent_accuracies))
            print(f"step={step + 1}  proto_loss={proto_losses[-1]:.4f}  proto_acc={proto_accuracies[-1]:.3f}")
            recent_losses, recent_accuracies = [], []

        if (step + 1) % logstep == 0:
            save_checkpoint(agent, os.path.join(dir, f"checkpoint_{step + 1}.pt"))
            np.save(os.path.join(dir, "episode_returns.npy"), np.array(episode_returns))
            np.save(os.path.join(dir, "proto_losses.npy"), np.array(proto_losses))
            np.save(os.path.join(dir, "proto_accuracies.npy"), np.array(proto_accuracies))
            if centroids is not None:
                np.save(os.path.join(dir, f"centroids_{step + 1}.npy"), centroids)

    return episode_returns


if __name__ == "__main__":
    train()
