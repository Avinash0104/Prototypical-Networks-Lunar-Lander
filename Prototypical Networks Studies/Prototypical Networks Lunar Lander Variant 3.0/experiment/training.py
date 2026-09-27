import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoder import Encoder
from models.sac import Actor, Critic
from experiment.environment import Environment, FrameStacker, FRAME_STACK, FRAME_CHANNELS
from experiment.buffer import ReplayBuffer
from models.predictor import Predictor
from models.prototype import PrototypeBank
from models.classifier import ClassifierHead
from experiment.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

class Agent:
    def __init__(self, action_dim=2, hidden_dim=128, z_dim=64, gamma=0.99, critic_target_tau=0.01, lr=1e-4,
                 device="cpu", arm="baseline", num_situations=4, num_prototypes=32, pred_dim=128, prototype_assignment_tau=0.1,
                 queue_size=128, topk=3, explore_coef=0.2, encoder_target_tau=0.05, encoder_update_frequency=2,
                 actor_update_frequency=2, critic_target_update_frequency=2, init_temperature=0.1, use_exploration_bonus=True):
        self.device = device
        self.gamma = gamma
        self.critic_target_tau = critic_target_tau
        self.encoder_target_tau = encoder_target_tau
        self.encoder_update_frequency = encoder_update_frequency
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.arm = arm
        self.topk = topk
        self.explore_coef = explore_coef
        self.use_exploration_bonus = use_exploration_bonus
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.lr = lr

        in_channels = FRAME_STACK * FRAME_CHANNELS
        self.encoder = Encoder(in_channels, z_dim).to(device)
        self.actor = Actor(z_dim, action_dim).to(device)
        self.critic = Critic(z_dim, action_dim).to(device)

        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.encoder_target = copy.deepcopy(self.encoder).to(device)
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        if arm == "prototype":
            self.predictor = Predictor(z_dim, pred_dim).to(device)
            self.prototype_bank = PrototypeBank(num_prototypes, z_dim, prototype_assignment_tau, queue_size=queue_size).to(device)
            self.proto_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()) + list(self.prototype_bank.parameters()), lr=lr)
        elif arm == "classifier":
            self.classifier_head = ClassifierHead(z_dim, hidden_dim, num_situations).to(device)
            self.classifier_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifier_head.parameters()), lr=lr)
        else:
            for p in self.encoder.parameters(): p.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.target_entropy = -action_dim
        self.init_temperature = init_temperature
        self.log_alpha = torch.full((1,), np.log(init_temperature), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def soft_update_target_critic(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.critic_target_tau * param.data + (1 - self.critic_target_tau) * target_param.data)

    def soft_update_target_encoder(self):
        for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            target_param.data.copy_(self.encoder_target_tau * param.data + (1 - self.encoder_target_tau) * target_param.data)

    def reset_critic(self):
        self.critic = Critic(self.z_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def reset_alpha(self):
        self.log_alpha = torch.full((1,), np.log(self.init_temperature), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def random_shift(self, obs, pad=4):
        b, c, h, w = obs.shape
        padded = F.pad(obs, (pad, pad, pad, pad), mode="replicate")

        shifted = torch.empty_like(obs)
        shift_x = torch.randint(0, 2 * pad + 1, (b,))
        shift_y = torch.randint(0, 2 * pad + 1, (b,))

        for i in range(b):
            shifted[i] = padded[i, :, shift_y[i]:shift_y[i] + h, shift_x[i]:shift_x[i] + w]
        return shifted

    def update_critic(self, obs, action, reward, next_obs, done, task_agnostic):
        with torch.no_grad():
            next_z_live = self.encoder(next_obs)
            next_action, next_log_prob = self.actor.sample(next_z_live)
            target_q1, target_q2 = self.critic_target(next_z_live, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            if self.arm == "prototype" and self.use_exploration_bonus:
                self.prototype_bank.update_queue(next_z_live)
                explore_reward = self.prototype_bank.compute_reward(next_z_live, topk=self.topk)
                used_reward = explore_reward if task_agnostic else reward + self.explore_coef * explore_reward
            else:
                used_reward = reward

            y = used_reward + self.gamma * (1 - done) * target_q

        z = self.encoder(obs)
        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

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

    def update_classifier_rep(self, obs_aug, situation):
        situation = situation.long().squeeze(-1)
        z = self.encoder(obs_aug)
        logits = self.classifier_head(z)
        classifier_loss =  F.cross_entropy(logits, situation)

        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        return classifier_loss

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

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss

    def update(self, batch, step, task_agnostic=False):
        obs, action, reward, next_obs, done, situation = batch
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        situation = situation.to(self.device)

        obs = self.random_shift(obs.float())
        next_obs = self.random_shift(next_obs.float())
        actor_loss, alpha_loss, aux_loss = None, None, None

        if self.arm == "prototype" and task_agnostic and step % self.encoder_update_frequency == 0:
            aux_loss = self.update_proto_rep(obs, next_obs)
            self.soft_update_target_encoder()
        elif self.arm == "classifier" and task_agnostic and step % self.encoder_update_frequency == 0:
            aux_loss = self.update_classifier_rep(obs, situation)

        critic_loss = self.update_critic(obs, action, reward, next_obs, done, task_agnostic)
        if step % self.actor_update_frequency == 0:
            actor_loss = self.update_actor(obs)
            alpha_loss = self.update_alpha(obs)
        if step % self.critic_target_update_frequency == 0:
            self.soft_update_target_critic()

        return {
            "critic_loss": critic_loss.item(),
            "aux_loss": aux_loss.item() if aux_loss is not None else None,
            "actor_loss": actor_loss.item() if actor_loss is not None else None,
            "alpha_loss": alpha_loss.item() if alpha_loss is not None else None,
            "alpha": self.alpha.item(),
            "task_agnostic": task_agnostic
        }

def train(num_steps=500_000, warmup_steps=1000, batch_size=256, buffer_capacity=200_000, seed=0,
          arm="baseline", logstep=50_000, result="results", resume=True, task_agnostic_frac=0.25,
          use_exploration_bonus=True):
    suffix = "_noexplore" if (arm == "prototype" and not use_exploration_bonus) else ""
    run_name = f"{arm}{suffix}_seed{seed}"
    dir = os.path.join(result, run_name)
    os.makedirs(dir, exist_ok=True)
    task_agnostic_steps = int(num_steps * task_agnostic_frac)

    env = Environment(seed=seed)
    agent = Agent(arm=arm, use_exploration_bonus=use_exploration_bonus)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    stacker = FrameStacker()

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

    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)
    steps_since_reset = 0
    episode_return = 0.0

    for step in range(start_step, num_steps):
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            stack_t = torch.from_numpy(stack).float().unsqueeze(0)
            with torch.no_grad():
                z = agent.encoder(stack_t)
                action, _ = agent.actor.sample(z)
                action = action.squeeze(0).numpy()

        situation = info["situation_label"]
        next_frame, reward, terminated, truncated, info = env.step(action)
        done = float(terminated)

        buffer.add(frame, action, reward, done, situation, steps_since_reset)

        frame = next_frame
        stack = stacker.step(next_frame)
        steps_since_reset = 0 if (terminated or truncated) else steps_since_reset + 1
        episode_return += reward

        if terminated or truncated:
            episode_returns.append(episode_return)
            if len(episode_returns) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                print(f"step={step}  episode={len(episode_returns)} avg_return(last10)={avg_return:.1f}")
            episode_return = 0.0
            frame, info = env.reset()
            stack = stacker.reset(frame)
            steps_since_reset = 0

        task_agnostic = step < task_agnostic_steps
        if arm == "prototype" and use_exploration_bonus and step == task_agnostic_steps:
            buffer.clear()
            agent.reset_critic()
            agent.reset_alpha()

        if step >= warmup_steps and buffer.size >= batch_size:
            agent.update(buffer.sample(batch_size), step, task_agnostic)

        if (step + 1) % logstep == 0:
            save_checkpoint(agent, step + 1, os.path.join(dir, f"checkpoint_{step + 1}.pt"))
            np.save(os.path.join(dir, "episode_returns.npy"), np.array(episode_returns))

    return episode_returns

if __name__ == "__main__":
    train()
