import os
import torch
import torch.nn.functional as F
import numpy as np

from experiment.training import Agent
from experiment.checkpoint import load_checkpoint
from experiment.environment import Environment, FrameStacker
from evaluation.probe import collect_embeddings, STATE_DIM_NAMES


def load_full_agent(checkpoint_path, device="cpu"):
    agent = Agent(device=device)
    load_checkpoint(agent, checkpoint_path)
    agent.encoder.eval()
    agent.encoder_target.eval()
    agent.predictor.eval()
    return agent


def evaluate_prototype_bank(agent, num_steps=5000, seed=4000):
    """
    Checks self-consistency with the model's own target branch: does the
    predicted next-step prototype (online encoder + predictor) match the
    prototype the target encoder assigns to the actual next frame? This is a
    training-objective generalization check, not a ground-truth accuracy
    measure -- see evaluate_cluster_state_coherence for that.
    """
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

    next_cluster_agreement = correct / num_steps

    num_prototypes = agent.prototype_bank.prototypes.shape[0]
    counts = np.bincount(assignments, minlength=num_prototypes)
    usage = counts / counts.sum()
    nonzero = usage[usage > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    utilization = entropy / np.log(num_prototypes)
    dead_prototypes = int((counts == 0).sum())

    return {
        "next_cluster_agreement": next_cluster_agreement,
        "utilization": utilization,
        "dead_prototypes": dead_prototypes,
        "num_prototypes": num_prototypes,
    }


def evaluate_augmentation_invariance(agent, num_steps=5000, batch_size=64, seed=5000):
    """
    Within a batch of distractor frames, is a frame's second random-shift
    augmentation closest (by cosine similarity) to its own first augmentation,
    rather than some other frame's? A harder, less luck-prone bar than
    matching coarse argmax cluster ids, especially with few live prototypes.
    """
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)

    stacks = np.zeros((num_steps,) + stack.shape, dtype=np.uint8)
    for i in range(num_steps):
        stacks[i] = stack
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)
        if terminated or truncated:
            frame, info = env.reset()
            stack = stacker.reset(frame)

    cos_sims = []
    correct_nn, total = 0, 0

    for start in range(0, num_steps, batch_size):
        batch = stacks[start:start + batch_size]
        if batch.shape[0] < 2:
            continue
        x = torch.from_numpy(batch).float()

        with torch.no_grad():
            aug1 = agent.random_shift(x)
            aug2 = agent.random_shift(x)

            z1 = F.normalize(agent.encoder(aug1), dim=-1)
            z2 = F.normalize(agent.encoder(aug2), dim=-1)

            sims = z1 @ z2.t()
            b = sims.shape[0]

            cos_sims.extend(sims.diagonal().tolist())

            nn_pred = sims.argmax(dim=-1)
            correct_nn += (nn_pred == torch.arange(b)).sum().item()
            total += b

    return {
        "nearest_neighbor_accuracy": correct_nn / total,
        "avg_cosine_similarity": float(np.mean(cos_sims)),
    }


def evaluate_cluster_state_coherence(agent, num_steps=5000, seed=7000):
    """
    Does the model's own (unsupervised) prototype assignment correspond to
    coherent regions of the *true* physical state -- independent of whether
    those regions match any hand-picked label like situation_label? Cluster
    membership is decided from z, but the quantity checked for coherence
    (true state) is completely independent of z, so this isn't circular the
    way assigning-and-measuring in the same space would be. High explained
    variance on a dimension means the model's self-discovered clusters
    correspond to real, distinguishable regimes along that dimension.
    """
    z, states, _, _ = collect_embeddings(agent, num_steps=num_steps, seed=seed)

    with torch.no_grad():
        z_norm = F.normalize(torch.from_numpy(z).float(), dim=-1)
        assignments = agent.prototype_bank(z_norm).argmax(dim=-1).numpy()

    values = states[:, :len(STATE_DIM_NAMES)]
    explained_variance = {}
    for i, name in enumerate(STATE_DIM_NAMES):
        col = values[:, i]
        total_var = col.var()
        if total_var == 0:
            explained_variance[name] = 0.0
            continue

        within = 0.0
        for cluster_id in np.unique(assignments):
            members = col[assignments == cluster_id]
            within += len(members) * members.var()
        within /= len(col)

        explained_variance[name] = float(1 - within / total_var)

    overall_coherence = float(np.mean(list(explained_variance.values())))
    return {"explained_variance": explained_variance, "overall_coherence": overall_coherence}


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=5000):
    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_full_agent(checkpoint_path)
        stats = evaluate_prototype_bank(agent, num_steps=num_steps)
        invariance = evaluate_augmentation_invariance(agent, num_steps=num_steps)
        coherence = evaluate_cluster_state_coherence(agent, num_steps=num_steps)

        row = {"seed": seed, **stats, **invariance, **coherence}
        rows.append(row)
        print(f"{run:20s} next_cluster_agreement={stats['next_cluster_agreement']:.3f} "
              f"utilization={stats['utilization']:.2f} "
              f"dead={stats['dead_prototypes']}/{stats['num_prototypes']} "
              f"aug_nn_acc={invariance['nearest_neighbor_accuracy']:.3f} "
              f"aug_cos_sim={invariance['avg_cosine_similarity']:.3f} "
              f"state_coherence={coherence['overall_coherence']:.3f}")
        for dim, val in coherence["explained_variance"].items():
            print(f"    {dim:10s} explained_variance={val:.3f}")

    agreements = [r["next_cluster_agreement"] for r in rows]
    utils = [r["utilization"] for r in rows]
    coherences = [r["overall_coherence"] for r in rows]
    print(f"\nnext_cluster_agreement avg={np.mean(agreements):.3f} +/- {np.std(agreements):.3f}   "
          f"utilization avg={np.mean(utils):.3f} +/- {np.std(utils):.3f}   "
          f"state_coherence avg={np.mean(coherences):.3f} +/- {np.std(coherences):.3f}")

    return rows


if __name__ == "__main__":
    main()
