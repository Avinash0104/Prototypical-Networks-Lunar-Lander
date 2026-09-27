import os
import torch

def load_pretrained_encoder(encoder, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder

def save_checkpoint(agent, path):
    checkpoint = {
        "encoder": agent.encoder.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "target": agent.target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "alpha_optimizer": agent.alpha_optimizer.state_dict(),
    }
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
    with torch.no_grad():
        agent.log_alpha.copy_(checkpoint["log_alpha"].to(agent.device))
        agent.log_alpha.clamp_(-10.0, 5.0)  # sanitize checkpoints saved before this bound existed
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])