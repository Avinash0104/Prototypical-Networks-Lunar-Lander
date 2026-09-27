import os
import torch
import torch.nn.functional as F
import numpy as np

from experiment.training import Agent
from experiment.checkpoint import load_checkpoint
from experiment.environment import Environment, FrameStacker


def load_full_agent(checkpoint_path, device="cpu"):
    agent = Agent(device=device)
    load_checkpoint(agent, checkpoint_path)
    agent.encoder.eval()
    agent.encoder_target.eval()
    agent.predictor.eval()
    return agent


def evaluate_prototype_bank(agent, num_steps=5000, seed=4000):
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    prev_stack = stacker.reset(frame)

    assignments = []
    correct = 0

    for _ in range(num_steps):
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)

        obs_t = torch.from_numpy(prev_stack).float().unsqueeze(0)
        next_obs_t = torch.from_numpy(stack).float().unsqueeze(0)

        with torch.no_grad():
            z_pred = F.normalize(agent.predictor(agent.encoder(obs_t)), dim=-1)
            predicted = agent.prototype_bank(z_pred).argmax(dim=-1)

            next_z = F.normalize(agent.encoder_target(next_obs_t), dim=-1)
            target = agent.prototype_bank(next_z).argmax(dim=-1)

        assignments.append(target.item())
        correct += (predicted == target).item()

        prev_stack = stack
        if terminated or truncated:
            frame, info = env.reset()
            prev_stack = stacker.reset(frame)

    predictive_accuracy = correct / num_steps

    num_prototypes = agent.prototype_bank.prototypes.shape[0]
    counts = np.bincount(assignments, minlength=num_prototypes)
    usage = counts / counts.sum()
    nonzero = usage[usage > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    utilization = entropy / np.log(num_prototypes)
    dead_prototypes = int((counts == 0).sum())

    return {
        "predictive_accuracy": predictive_accuracy,
        "utilization": utilization,
        "dead_prototypes": dead_prototypes,
        "num_prototypes": num_prototypes,
    }


def evaluate_augmentation_invariance(agent, num_steps=5000, seed=5000):
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)

    same_cluster = 0
    cos_sims = []

    for _ in range(num_steps):
        stack_t = torch.from_numpy(stack).float().unsqueeze(0)

        with torch.no_grad():
            aug1 = agent.random_shift(stack_t)
            aug2 = agent.random_shift(stack_t)

            z1 = F.normalize(agent.encoder(aug1), dim=-1)
            z2 = F.normalize(agent.encoder(aug2), dim=-1)

            cluster1 = agent.prototype_bank(z1).argmax(dim=-1)
            cluster2 = agent.prototype_bank(z2).argmax(dim=-1)

            same_cluster += (cluster1 == cluster2).item()
            cos_sims.append((z1 * z2).sum().item())

        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)
        if terminated or truncated:
            frame, info = env.reset()
            stack = stacker.reset(frame)

    return {
        "same_cluster_rate": same_cluster / num_steps,
        "avg_cosine_similarity": float(np.mean(cos_sims)),
    }


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=5000):
    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_full_agent(checkpoint_path)
        stats = evaluate_prototype_bank(agent, num_steps=num_steps)
        invariance = evaluate_augmentation_invariance(agent, num_steps=num_steps)

        row = {"seed": seed, **stats, **invariance}
        rows.append(row)
        print(f"{run:20s} pred_acc={stats['predictive_accuracy']:.3f} "
              f"utilization={stats['utilization']:.2f} "
              f"dead={stats['dead_prototypes']}/{stats['num_prototypes']} "
              f"aug_same_cluster={invariance['same_cluster_rate']:.3f} "
              f"aug_cos_sim={invariance['avg_cosine_similarity']:.3f}")

    pred_accs = [r["predictive_accuracy"] for r in rows]
    utils = [r["utilization"] for r in rows]
    print(f"\npred_acc avg={np.mean(pred_accs):.3f} +/- {np.std(pred_accs):.3f}   "
          f"utilization avg={np.mean(utils):.3f} +/- {np.std(utils):.3f}")

    return rows


if __name__ == "__main__":
    main()
