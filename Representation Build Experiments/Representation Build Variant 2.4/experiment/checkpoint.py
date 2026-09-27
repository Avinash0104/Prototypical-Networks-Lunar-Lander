import os
import torch

def save_checkpoint(agent, path):
    checkpoint = {
        "encoder": agent.encoder.state_dict(),
        "encoder_target": agent.encoder_target.state_dict(),
        "proto_optimizer": agent.proto_optimizer.state_dict(),
    }
    # write-then-rename so a crash mid-save can never leave a truncated checkpoint_N.pt
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

def find_latest_checkpoint(dir):
    if not os.path.isdir(dir): return None, 0
    checkpoints = [f for f in os.listdir(dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not checkpoints: return None, 0
    steps = [int(f[len("checkpoint_"):-len(".pt")]) for f in checkpoints]
    latest_step = max(steps)
    return os.path.join(dir, f"checkpoint_{latest_step}.pt"), latest_step

def find_all_checkpoints(dir):
    if not os.path.isdir(dir): return []
    checkpoints = [f for f in os.listdir(dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    return sorted(int(f[len("checkpoint_"):-len(".pt")]) for f in checkpoints)

def load_checkpoint(agent, path):
    checkpoint = torch.load(path, map_location=agent.device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.encoder_target.load_state_dict(checkpoint["encoder_target"])
    agent.proto_optimizer.load_state_dict(checkpoint["proto_optimizer"])
