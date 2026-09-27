import os
import torch

def save_checkpoint(agent, step, path):
    checkpoint = {
        "step": step,
        "encoder": agent.encoder.state_dict(),
        "encoder_target": agent.encoder_target.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "alpha_optimizer": agent.alpha_optimizer.state_dict(),
        "arm": agent.arm,
    }
    if agent.arm == "prototype":
        checkpoint["predictor"] = agent.predictor.state_dict()
        checkpoint["prototype_bank"] = agent.prototype_bank.state_dict()
        checkpoint["proto_optimizer"] = agent.proto_optimizer.state_dict()
    elif agent.arm == "classifier":
        checkpoint["classifier_head"] = agent.classifier_head.state_dict()
        checkpoint["classifier_optimizer"] = agent.classifier_optimizer.state_dict()
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
    agent.encoder_target.load_state_dict(checkpoint.get("encoder_target", checkpoint["encoder"]))
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.critic_target.load_state_dict(checkpoint["critic_target"])
    if "critic_optimizer" in checkpoint:
        agent.log_alpha.data.copy_(checkpoint["log_alpha"].to(agent.device))
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
    if agent.arm == "prototype":
        agent.predictor.load_state_dict(checkpoint["predictor"])
        agent.prototype_bank.load_state_dict(checkpoint["prototype_bank"])
        if "proto_optimizer" in checkpoint:
            agent.proto_optimizer.load_state_dict(checkpoint["proto_optimizer"])
    elif agent.arm == "classifier":
        agent.classifier_head.load_state_dict(checkpoint["classifier_head"])
        if "classifier_optimizer" in checkpoint:
            agent.classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])