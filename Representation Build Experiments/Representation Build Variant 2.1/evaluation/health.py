import os
import numpy as np
import torch

from experiment.training import Agent
from experiment.checkpoint import load_checkpoint
from experiment.environment import Environment
from experiment.buffer import ReplayBuffer
from experiment.clustering import fit_centroids, assign_clusters, diff_feature
from models.prototype import prototypical_loss


def load_agent(checkpoint_path, device="cpu"):
    agent = Agent(device=device)
    load_checkpoint(agent, checkpoint_path)
    agent.encoder.eval()
    agent.encoder_target.eval()
    return agent


def _collect_held_out_buffer(num_steps, seed):
    env = Environment(seed=seed)
    buffer = ReplayBuffer(capacity=num_steps)
    frame, info = env.reset(seed=seed)
    steps_since_reset = 0

    for _ in range(num_steps):
        action = env.action_space.sample()
        next_frame, reward, terminated, truncated, info = env.step(action)
        buffer.add(frame, steps_since_reset)
        frame = next_frame
        steps_since_reset = 0 if (terminated or truncated) else steps_since_reset + 1
        if terminated or truncated:
            frame, info = env.reset()
            steps_since_reset = 0

    return buffer


def _run_matching_batches(agent, buffer, centroids, num_classes, num_support, num_query,
                           candidate_pool, pool_size, num_batches):
    losses, accs = [], []

    for _ in range(num_batches):
        batch = buffer.sample_by_cluster(centroids, n_classes=num_classes, num_support=num_support,
                                          num_query=num_query, candidate_pool=candidate_pool, pool_size=pool_size)
        if batch is None:
            continue

        support_next_obs, query_next_obs = batch
        support_next_obs = support_next_obs.to(agent.device).float()
        query_next_obs = query_next_obs.to(agent.device).float()

        C, K = support_next_obs.shape[:2]
        Q = query_next_obs.shape[1]

        with torch.no_grad():
            support_flat = support_next_obs.view(-1, *support_next_obs.shape[2:])
            z_support = agent.encoder(support_flat)
            support_y = torch.arange(C, device=agent.device).repeat_interleave(K)

            query_flat = query_next_obs.view(-1, *query_next_obs.shape[2:])
            z_query = agent.encoder_target(query_flat)
            query_y = torch.arange(C, device=agent.device).repeat_interleave(Q)

            loss, acc = prototypical_loss(z_support, support_y, z_query, query_y, num_classes=C)

        losses.append(loss.item())
        accs.append(acc.item())

    if not losses:
        return {"held_out_loss": None, "held_out_accuracy": None, "num_batches_used": 0}

    return {
        "held_out_loss": float(np.mean(losses)),
        "held_out_accuracy": float(np.mean(accs)),
        "num_batches_used": len(losses),
    }


def evaluate_motion_matching_fresh(agent, buffer, num_classes=16, pool_size=8, num_support=4, num_query=4,
                                    candidate_pool=1000, centroid_sample_size=4000, num_batches=20, seed=None):
    """
    Fits brand-new centroids on the held-out buffer itself, then replays the exact
    training task (cluster -> support/query -> prototypical_loss) with no gradient.
    Tests general competence at the task pattern, independent of any specific
    centroids from training.
    """
    centroids = fit_centroids(buffer, sample_size=centroid_sample_size, k=num_classes, seed=seed, pool_size=pool_size)
    return _run_matching_batches(agent, buffer, centroids, num_classes, num_support, num_query,
                                  candidate_pool, pool_size, num_batches)


def evaluate_motion_matching_persisted(agent, buffer, centroids, num_support=4, num_query=4,
                                        candidate_pool=1000, pool_size=8, num_batches=20):
    """
    Reuses the exact centroids saved during training, then replays the same task on
    held-out data. Tests whether the specific motion categories the model was
    actually trained on generalize to unseen transitions.
    """
    num_classes = centroids.shape[0]
    return _run_matching_batches(agent, buffer, centroids, num_classes, num_support, num_query,
                                  candidate_pool, pool_size, num_batches)


def evaluate_cluster_separation(agent, buffer, centroids, pool_size=8, num_samples=2000, seed=None):
    """
    For held-out transitions labeled by the given centroids: average embedding-space
    distance within a cluster vs. between different clusters. A finer-grained view of
    the same thing held_out_accuracy measures coarsely.
    """
    index = buffer._random_valid_indices(min(num_samples, buffer.size))
    obs, next_obs = buffer.gather(index)

    feats = diff_feature(torch.from_numpy(obs), torch.from_numpy(next_obs), pool_size=pool_size).numpy()
    assignments = assign_clusters(feats, centroids)

    with torch.no_grad():
        z = agent.encoder(torch.from_numpy(next_obs).float().to(agent.device))
    z = z.cpu().numpy()

    rng = np.random.default_rng(seed)
    n = len(assignments)
    num_pairs = min(20000, n * (n - 1) // 2)
    pair_idx = rng.integers(0, n, size=(num_pairs, 2))

    within, between = [], []
    for i, j in pair_idx:
        if i == j:
            continue
        dist = float(np.linalg.norm(z[i] - z[j]))
        if assignments[i] == assignments[j]:
            within.append(dist)
        else:
            between.append(dist)

    within_mean = float(np.mean(within)) if within else None
    between_mean = float(np.mean(between)) if between else None
    ratio = (between_mean / within_mean) if (within_mean and between_mean and within_mean > 0) else None

    return {
        "avg_within_cluster_dist": within_mean,
        "avg_between_cluster_dist": between_mean,
        "separation_ratio": ratio,
    }


def evaluate_cluster_balance(centroids, buffer, pool_size=8, num_samples=4000, seed=None):
    """
    How evenly the discovered motion clusters are populated on held-out data.
    Same entropy-based utilization formula used for proto-RL's prototype bank,
    adapted to k-means cluster assignment counts.
    """
    index = buffer._random_valid_indices(min(num_samples, buffer.size))
    obs, next_obs = buffer.gather(index)

    feats = diff_feature(torch.from_numpy(obs), torch.from_numpy(next_obs), pool_size=pool_size).numpy()
    assignments = assign_clusters(feats, centroids)

    num_classes = centroids.shape[0]
    counts = np.bincount(assignments, minlength=num_classes)
    usage = counts / counts.sum()
    nonzero = usage[usage > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    utilization = entropy / np.log(num_classes)
    dead_clusters = int((counts == 0).sum())

    return {
        "utilization": float(utilization),
        "dead_clusters": dead_clusters,
        "num_classes": num_classes,
    }


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=5000,
         num_classes=16, pool_size=8, num_support=4, num_query=4,
         candidate_pool=1000, centroid_sample_size=4000, num_batches=20):
    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        centroids_path = os.path.join(results_dir, run, f"centroids_{checkpoint_step}.npy")

        agent = load_agent(checkpoint_path)
        held_out_buffer = _collect_held_out_buffer(num_steps, seed=10_000 + seed)

        fresh = evaluate_motion_matching_fresh(
            agent, held_out_buffer, num_classes=num_classes, pool_size=pool_size,
            num_support=num_support, num_query=num_query, candidate_pool=candidate_pool,
            centroid_sample_size=centroid_sample_size, num_batches=num_batches, seed=seed)

        row = {
            "seed": seed,
            "fresh_held_out_loss": fresh["held_out_loss"],
            "fresh_held_out_accuracy": fresh["held_out_accuracy"],
        }

        if os.path.exists(centroids_path):
            centroids = np.load(centroids_path)

            persisted = evaluate_motion_matching_persisted(
                agent, held_out_buffer, centroids, num_support=num_support, num_query=num_query,
                candidate_pool=candidate_pool, pool_size=pool_size, num_batches=num_batches)
            separation = evaluate_cluster_separation(agent, held_out_buffer, centroids, pool_size=pool_size, seed=seed)
            balance = evaluate_cluster_balance(centroids, held_out_buffer, pool_size=pool_size, seed=seed)

            row.update({
                "persisted_held_out_loss": persisted["held_out_loss"],
                "persisted_held_out_accuracy": persisted["held_out_accuracy"],
                "separation_ratio": separation["separation_ratio"],
                "utilization": balance["utilization"],
                "dead_clusters": balance["dead_clusters"],
            })
        else:
            print(f"[{run}] no centroids_{checkpoint_step}.npy found -- skipping persisted-centroid checks")

        rows.append(row)

        line = (f"{run:20s} "
                f"fresh_loss={row['fresh_held_out_loss']:.3f} fresh_acc={row['fresh_held_out_accuracy']:.3f}")
        if "persisted_held_out_loss" in row:
            line += (f" persisted_loss={row['persisted_held_out_loss']:.3f} "
                     f"persisted_acc={row['persisted_held_out_accuracy']:.3f} "
                     f"separation={row['separation_ratio']:.2f} "
                     f"utilization={row['utilization']:.2f} "
                     f"dead={row['dead_clusters']}/{centroids.shape[0]}")
        print(line)

    return rows


if __name__ == "__main__":
    main()
