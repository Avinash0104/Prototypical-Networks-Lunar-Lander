import os
import numpy as np
import torch

from experiment.training import Agent
from experiment.checkpoint import load_checkpoint
from experiment.environment import Environment, FrameStacker
from models.prototype import prototypical_loss


def load_agent(checkpoint_path, device="cpu"):
    agent = Agent(device=device)
    load_checkpoint(agent, checkpoint_path)
    agent.encoder.eval()
    agent.encoder_target.eval()
    return agent


def _collect_frames(num_frames, seed):
    env = Environment(seed=seed)
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)

    frames = np.zeros((num_frames,) + stack.shape, dtype=np.uint8)
    for i in range(num_frames):
        frames[i] = stack
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)
        if terminated or truncated:
            frame, info = env.reset()
            stack = stacker.reset(frame)
    return frames


def evaluate_augmentation_invariance(agent, num_steps=2000, batch_size=64, seed=5000):
    """
    Checks the property the whole algorithm leans on: two random-shift
    augmentations of the *same* frame should encode to nearby vectors,
    while augmentations of *different* frames should not. Measured with
    the same metric prototypical_loss actually trains on -- raw squared
    Euclidean distance on unnormalized embeddings, not cosine similarity.
    Reported as a nearest-neighbour accuracy: for each frame's first
    augmentation, is its second augmentation the closest match (smallest
    distance) among the batch, rather than some other frame's augmentation?
    """
    frames = _collect_frames(num_steps, seed)

    same_dists, other_dists = [], []
    correct_nn, total = 0, 0

    for start in range(0, num_steps, batch_size):
        batch = frames[start:start + batch_size]
        if batch.shape[0] < 2:
            continue
        x = torch.from_numpy(batch).float()

        with torch.no_grad():
            aug1 = agent.random_shift(x)
            aug2 = agent.random_shift(x)
            z1 = agent.encoder(aug1)
            z2 = agent.encoder(aug2)

            dists = torch.cdist(z1, z2, p=2).pow(2)
            b = dists.shape[0]

            same_dists.extend(dists.diagonal().tolist())
            mask = ~torch.eye(b, dtype=torch.bool)
            other_dists.extend(dists[mask].tolist())

            nn_pred = dists.argmin(dim=-1)
            correct_nn += (nn_pred == torch.arange(b)).sum().item()
            total += b

    return {
        "avg_same_frame_sq_dist": float(np.mean(same_dists)),
        "avg_other_frame_sq_dist": float(np.mean(other_dists)),
        "nearest_neighbor_accuracy": correct_nn / total,
    }


def evaluate_prototype_matching(agent, num_batches=20, batch_size=256, seed=6000):
    """
    Re-runs the exact obs-support / next_obs-query matching task the
    training loss optimizes, but on a fresh held-out rollout (not the
    training buffer) with no gradient step -- i.e. does the algorithm's
    own objective generalize past the data it was trained on?
    """
    frames = _collect_frames(num_batches * batch_size + 1, seed)
    K = agent.num_support

    losses, accs = [], []
    for _ in range(num_batches):
        idx = np.random.randint(0, len(frames) - 1, size=batch_size)
        obs = torch.from_numpy(frames[idx]).float()
        next_obs = torch.from_numpy(frames[idx + 1]).float()

        with torch.no_grad():
            B = obs.shape[0]
            support_in = agent.random_shift(obs.repeat_interleave(K, dim=0))
            z_support = agent.encoder(support_in)
            support_y = torch.arange(B).repeat_interleave(K)

            query_in = agent.random_shift(next_obs)
            z_query = agent.encoder_target(query_in)
            query_y = torch.arange(B)

            loss, acc = prototypical_loss(z_support, support_y, z_query, query_y, num_classes=B)

        losses.append(loss.item())
        accs.append(acc.item())

    return {
        "held_out_loss": float(np.mean(losses)),
        "held_out_accuracy": float(np.mean(accs)),
    }


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=2000):
    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_agent(checkpoint_path)

        invariance = evaluate_augmentation_invariance(agent, num_steps=num_steps)
        matching = evaluate_prototype_matching(agent)

        row = {"seed": seed, **invariance, **matching}
        rows.append(row)
        print(f"{run:20s} "
              f"same_frame_sqdist={invariance['avg_same_frame_sq_dist']:.3f} "
              f"other_frame_sqdist={invariance['avg_other_frame_sq_dist']:.3f} "
              f"aug_nn_acc={invariance['nearest_neighbor_accuracy']:.3f} "
              f"held_out_loss={matching['held_out_loss']:.3f} "
              f"held_out_acc={matching['held_out_accuracy']:.3f}")

    nn_accs = [r["nearest_neighbor_accuracy"] for r in rows]
    held_accs = [r["held_out_accuracy"] for r in rows]
    print(f"\naug_nn_acc avg={np.mean(nn_accs):.3f} +/- {np.std(nn_accs):.3f}   "
          f"held_out_acc avg={np.mean(held_accs):.3f} +/- {np.std(held_accs):.3f}")

    return rows


if __name__ == "__main__":
    main()
