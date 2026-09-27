import os
import torch

from models.encoder import Encoder
from models.sac import Actor, Critic
from experiment.environment import Environment
from experiment.buffer import ReplayBuffer
from models.projection import ProjectionHead
from models.prototype import PrototypeBank
from models.classifier import ClassifierHead

def save_checkpoint(agent, step, path):
    checkpoint = {
        "step": step,
        "encoder": agent.encoder.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "target": agent.target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "alpha_optimizer": agent.alpha_optimizer.state_dict(),
        "arm": agent.arm,
    }
    if agent.arm == "prototype":
        checkpoint["projection_head"] = agent.projection_head.state_dict()
        checkpoint["prototype_bank"] = agent.prototype_bank.state_dict()
    elif agent.arm == "classifier":
        checkpoint["classifier_head"] = agent.classifier_head.state_dict()
    torch.save(checkpoint, path)

def find_latest_checkpoint(dir):
    if not os.path.isdir(dir): return None, 0
    checkpoints = [f for f in os.listdir(dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not checkpoints: return None, 0
    steps = [int(f[len("checkpoint_"):-len(".pt")]) for f in checkpoints]
    latest_step = max(steps)
    return os.path.join(dir, f"checkpoint_{latest_step}.pt"), latest_step

def load_checkpoint(agent, path):
    checkpoint = torch.load(path, map_location=agent.device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.target.load_state_dict(checkpoint["target"])
    if "critic_optimizer" in checkpoint:
        agent.log_alpha.data.copy_(checkpoint["log_alpha"].to(agent.device))
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
    if agent.arm == "prototype":
        agent.projection_head.load_state_dict(checkpoint["projection_head"])
        agent.prototype_bank.load_state_dict(checkpoint["prototype_bank"])
    elif agent.arm == "classifier":
        agent.classifier_head.load_state_dict(checkpoint["classifier_head"])