import os
import torch
import numpy as np

def save_checkpoint(agent, path):
    checkpoint = {
        "encoder": agent.encoder.state_dict(),
        "encoder_target": agent.encoder_target.state_dict(),
        "predictor": agent.predictor.state_dict(),
        "prototype_bank": agent.prototype_bank.state_dict(),
        "proto_optimizer": agent.proto_optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

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
    agent.predictor.load_state_dict(checkpoint["predictor"])
    agent.prototype_bank.load_state_dict(checkpoint["prototype_bank"])
    agent.proto_optimizer.load_state_dict(checkpoint["proto_optimizer"])

def save_replay_buffer(buffer, dir):
    # rolling file overwritten every save, not versioned per step - the buffer is
    # multiple GB, so keeping one copy per checkpoint step would blow up disk usage
    path = os.path.join(dir, "replay_buffer.npz")
    tmp_path = path + ".tmp.npz"
    np.savez(
        tmp_path,
        frames=buffer.frames,
        steps_since_reset=buffer.steps_since_reset,
        dones=buffer.dones,
        ptr=buffer.ptr,
        size=buffer.size,
    )
    os.replace(tmp_path, path)

def load_replay_buffer(buffer, dir):
    path = os.path.join(dir, "replay_buffer.npz")
    if not os.path.exists(path):
        return False
    data = np.load(path)
    buffer.frames[:] = data["frames"]
    buffer.steps_since_reset[:] = data["steps_since_reset"]
    buffer.dones[:] = data["dones"]
    buffer.ptr = int(data["ptr"])
    buffer.size = int(data["size"])
    return True
