import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoder import Encoder
from models.sac import Actor, Critic
from experiment.environment import Environment
from experiment.buffer import ReplayBuffer
from models.projection import ProjectionHead
from models.prototype import PrototypeBank
from models.classifier import ClassifierHead
from experiment.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

class Agent:
    def __init__(self, obs_dim=136, action_dim=2, hidden_dim=128, z_dim=64, gamma=0.99, tau=0.005, lr=3e-4, device="cpu", arm="baseline", num_situations=4, u_dim=32, proto_tau=0.1, noise_dims=128, noise_clip=3.0, collapse_weight=0.7):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.arm = arm
        self.noise_dims = noise_dims
        self.noise_clip = noise_clip
        self.collapse_weight = collapse_weight

        self.encoder = Encoder(obs_dim, hidden_dim, z_dim).to(device)
        self.actor = Actor(z_dim, action_dim).to(device)
        self.critic = Critic(z_dim, action_dim).to(device)

        self.target = copy.deepcopy(self.critic).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        aux_params = []
        if arm == "prototype":
            self.projection_head = ProjectionHead(z_dim, u_dim).to(device)
            self.prototype_bank = PrototypeBank(num_situations, u_dim, proto_tau).to(device)
            aux_params = list(self.projection_head.parameters()) + list(self.prototype_bank.parameters())
            # self.class_prior = torch.ones(num_situations, device=device) / num_situations
            # self.class_prior_momentum = 0.99
        elif arm == "classifier":
            self.classifier_head = ClassifierHead(z_dim, hidden_dim, num_situations).to(device)
            aux_params = list(self.classifier_head.parameters())

        self.critic_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.critic.parameters()) + aux_params, lr=lr)
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

    def update_critic(self, obs, action, reward, next_obs, done, situation, true_obs, lambda_t):
        with torch.no_grad():
            next_z = self.encoder(next_obs)
            next_action, next_log_prob = self.actor.sample(next_z)
            target_q1, target_q2 = self.target(next_z, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = reward + self.gamma * (1 - done) * target_q
        
        z = self.encoder(obs)
        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        aux_loss = self.update_aux(z, situation, true_obs)
        total_loss = critic_loss + lambda_t * aux_loss

        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.critic_optimizer.step()

        if self.arm == "prototype":
            self.prototype_bank.renormalize()

        return critic_loss, aux_loss

    def update_actor(self, obs):
        z = self.encoder(obs).detach()
        action, log_prob = self.actor.sample(z)

        q1, q2 = self.critic(z, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss

    def update_alpha(self, obs):
        z = self.encoder(obs).detach()
        action, log_prob = self.actor.sample(z)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss

    def update_aux(self, z, situation, true_obs=None):
        situation = situation.long().squeeze(-1)
        if self.arm == "prototype":
            u = self.projection_head(z)
            logits1 = self.prototype_bank(u)
            label_loss =  F.cross_entropy(logits1, situation)

            fresh_noise = torch.randn(true_obs.shape[0], self.noise_dims, device=self.device).clamp(-self.noise_clip, self.noise_clip)
            obs2 = torch.cat([true_obs, fresh_noise], dim=-1)

            z2 = self.encoder(obs2)
            u2 = self.projection_head(z2)
            logits2 = self.prototype_bank(u2)

            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)

            consistency_loss = -(probs2.detach() * log_probs1).sum(dim=-1).mean() - (probs1.detach() * log_probs2).sum(dim=-1).mean()
            repulsion_penalty = self.prototype_repulsion()

            return label_loss + consistency_loss + self.collapse_weight * repulsion_penalty

        elif self.arm == "classifier":
            logits = self.classifier_head(z)
            return F.cross_entropy(logits, situation)
        else:
            return torch.tensor(0.0, device=z.device)

    def prototype_repulsion(self):
        proto = self.prototype_bank.prototypes
        k = proto.shape[0]
        sim_matrix = proto @ proto.t()
        off_diag_mask = 1 - torch.eye(k, device=proto.device)
        return (sim_matrix * off_diag_mask).pow(2).sum() / (k * (k - 1))

    def update(self, batch, step):
        obs, action, reward, next_obs, done, situation, true_obs = batch
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        situation = situation.to(self.device)
        true_obs = true_obs.to(self.device)

        lambda_t = lambda_schedule(step)
        critic_loss, aux_loss = self.update_critic(obs, action, reward, next_obs, done, situation, true_obs, lambda_t)
        actor_loss = self.update_actor(obs)
        alpha_loss = self.update_alpha(obs)
        self.soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "aux_loss": aux_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "lambda_t": lambda_t
        }

def train(num_steps=300_000, warmup_steps=1000, batch_size=256, buffer_capacity=1_000_000, seed=0, arm="baseline", logstep=50_000, result="results", resume=True):
    run_name = f"{arm}_seed{seed}"
    dir = os.path.join(result, run_name)
    os.makedirs(dir, exist_ok=True)

    env = Environment(seed=seed)
    agent = Agent(arm=arm)
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
            print(f"[{run_name}] resuming from step {start_step} ({len(episode_returns)} episodes so far)", flush=True)

    obs, info = env.reset(seed=seed)
    episode_return = 0.0

    for step in range(start_step, num_steps):
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                z = agent.encoder(obs_t)
                action, _ = agent.actor.sample(z)
                action = action.squeeze(0).numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = float(terminated)
        buffer.add(obs, action, reward, next_obs, done, info["situation_label"], info["true_obs"])

        episode_return += reward
        obs = next_obs

        if terminated or truncated:
            episode_returns.append(episode_return)
            if len(episode_returns) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                print(f"step={step}  episode={len(episode_returns)} avg_return(last10)={avg_return:.1f}")
            episode_return = 0.0
            obs, info = env.reset()

        if step >= warmup_steps and buffer.size >= batch_size:
            agent.update(buffer.sample(batch_size), step)

        if (step + 1) % logstep == 0:
            save_checkpoint(agent, step + 1, os.path.join(dir, f"checkpoint_{step + 1}.pt"))
            np.save(os.path.join(dir, "episode_returns.npy"), np.array(episode_returns))

    return episode_returns


def lambda_schedule(step, warmup_steps=10_000, ramp_steps=10_000, final_lambda=1.0):
    if step < warmup_steps:
        return 0.0
    elif step < warmup_steps + ramp_steps:
        progress = (step - warmup_steps) / ramp_steps
        return progress * final_lambda
    else:
        return final_lambda

if __name__ == "__main__":
    train()





    