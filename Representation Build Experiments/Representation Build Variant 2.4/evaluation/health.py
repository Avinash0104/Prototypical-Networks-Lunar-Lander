import os
import json
import numpy as np
import torch

from experiment.training import Agent
from experiment.checkpoint import load_checkpoint
from experiment.environment import Environment
from experiment.buffer import ReplayBuffer
from experiment.clustering import fit_centroids, assign_clusters, frame_feature, POOL_SIZE
from models.prototype import prototypical_loss
from evaluation.probe import STATE_DIM_NAMES


def load_agent(checkpoint_path, device="cpu"):
    agent = Agent(device=device)
    load_checkpoint(agent, checkpoint_path)
    agent.encoder.eval()
    agent.encoder_target.eval()
    return agent


def random_agent(device="cpu"):
    agent = Agent(device=device)
    agent.encoder.eval()
    agent.encoder_target.eval()
    return agent


def _collect_held_out_buffer(num_steps, seed):
    env = Environment()
    env.action_space.seed(seed)
    # capacity headroom covers the worst case of every single step terminating an
    # episode, which adds one extra (terminal-frame) buffer entry per step -- see below
    buffer = ReplayBuffer(capacity=num_steps * 2)
    states = []
    frame, info = env.reset(seed=seed)
    steps_since_reset = 0

    for _ in range(num_steps):
        buffer.add(frame, steps_since_reset)
        states.append(info["true_state"])  # describes `frame`, from the same reset()/step() call
        steps_since_reset += 1
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # the terminal frame would otherwise be overwritten by reset()
            buffer.add(frame, steps_since_reset)
            states.append(info["true_state"])
            frame, info = env.reset()
            steps_since_reset = 0

    return buffer, np.array(states, dtype=np.float64)


def _run_matching_batches(agent, buffer, centroids, num_classes, num_support, num_query,
                           candidate_pool, pool_size, num_batches):
    losses, accs = [], []

    for _ in range(num_batches):
        batch = buffer.sample_by_cluster(centroids, n_classes=num_classes, num_support=num_support,
                                          num_query=num_query, candidate_pool=candidate_pool, pool_size=pool_size)
        if batch is None:
            continue

        support_obs, query_obs = batch
        support_obs = support_obs.to(agent.device).float()
        query_obs = query_obs.to(agent.device).float()

        C, K = support_obs.shape[:2]
        Q = query_obs.shape[1]

        with torch.no_grad():
            support_flat = support_obs.reshape(-1, *support_obs.shape[2:])
            z_support = agent.encoder(support_flat)
            support_y = torch.arange(C, device=agent.device).repeat_interleave(K)

            query_flat = query_obs.reshape(-1, *query_obs.shape[2:])
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


def evaluate_frame_matching_fresh(agent, buffer, num_classes=16, pool_size=POOL_SIZE, num_support=4, num_query=4,
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


def evaluate_frame_matching_persisted(agent, buffer, centroids, num_support=4, num_query=4,
                                        candidate_pool=1000, pool_size=POOL_SIZE, num_batches=20):
    """
    Reuses the exact centroids saved during training, then replays the same task on
    held-out data. Tests whether the specific frame categories the model was
    actually trained on generalize to unseen frames.
    """
    num_classes = centroids.shape[0]
    return _run_matching_batches(agent, buffer, centroids, num_classes, num_support, num_query,
                                  candidate_pool, pool_size, num_batches)


def evaluate_cluster_separation(agent, buffer, centroids, pool_size=POOL_SIZE, num_samples=2000, seed=None):
    """
    For held-out frames labeled by the given centroids: average embedding-space
    distance within a cluster vs. between different clusters. A finer-grained view of
    the same thing held_out_accuracy measures coarsely.
    """
    index = buffer.sample_indices(num_samples)
    stacks = torch.from_numpy(buffer.gather(index))

    feats = frame_feature(stacks, pool_size=pool_size).numpy()
    assignments = assign_clusters(feats, centroids)

    with torch.no_grad():
        z = agent.encoder(stacks.float().to(agent.device))
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


def evaluate_cluster_balance(centroids, buffer, pool_size=POOL_SIZE, num_samples=4000, seed=None):
    """
    How evenly the discovered frame clusters are populated on held-out data.
    Same entropy-based utilization formula used for proto-RL's prototype bank,
    adapted to k-means cluster assignment counts.
    """
    index = buffer.sample_indices(num_samples)
    stacks = torch.from_numpy(buffer.gather(index))

    feats = frame_feature(stacks, pool_size=pool_size).numpy()
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


def evaluate_cluster_state_coherence(centroids, buffer, states, pool_size=POOL_SIZE, num_samples=4000, seed=None):
    """
    Does the unsupervised frame-cluster assignment correspond to coherent regions
    of the *true* physical state of that same frame -- independent of any
    hand-picked label like situation_label? Cluster membership is decided from
    pooled pixels, but the quantity checked for coherence (true state) is
    completely independent of that, so this isn't circular the way
    assigning-and-measuring in the same space would be. High explained variance on
    a dimension means the discovered frame clusters correspond to real,
    distinguishable states along that dimension -- this is the only health.py
    check that validates the clustering itself against ground truth rather than
    against the model's own training signal. `states[index]` is the state of the
    latest frame in each stack.
    """
    index = buffer.sample_indices(num_samples)
    stacks = torch.from_numpy(buffer.gather(index))

    feats = frame_feature(stacks, pool_size=pool_size).numpy()
    assignments = assign_clusters(feats, centroids)

    values = states[index][:, :len(STATE_DIM_NAMES)]
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


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=5000,
         num_classes=16, pool_size=POOL_SIZE, num_support=4, num_query=4,
         candidate_pool=1000, centroid_sample_size=4000, num_batches=20, num_floor_draws=3):
    # Untrained encoders, shared across seeds: every metric below is also computed
    # against these floors (averaged over several fresh random inits, since a
    # single random init is a noisy read on "what does architecture alone give
    # you") so a score can be judged as "better than chance/random init" rather
    # than read as an unanchored number.
    floor_agents = [random_agent() for _ in range(num_floor_draws)]

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        centroids_path = os.path.join(results_dir, run, f"centroids_{checkpoint_step}.npy")

        agent = load_agent(checkpoint_path)
        held_out_buffer, states = _collect_held_out_buffer(num_steps, seed=10_000 + seed)

        fresh = evaluate_frame_matching_fresh(
            agent, held_out_buffer, num_classes=num_classes, pool_size=pool_size,
            num_support=num_support, num_query=num_query, candidate_pool=candidate_pool,
            centroid_sample_size=centroid_sample_size, num_batches=num_batches, seed=seed)
        floor_fresh_acc = float(np.mean([
            evaluate_frame_matching_fresh(
                fa, held_out_buffer, num_classes=num_classes, pool_size=pool_size,
                num_support=num_support, num_query=num_query, candidate_pool=candidate_pool,
                centroid_sample_size=centroid_sample_size, num_batches=num_batches, seed=seed
            )["held_out_accuracy"] for fa in floor_agents
        ]))

        row = {
            "seed": seed,
            "fresh_held_out_loss": fresh["held_out_loss"],
            "fresh_held_out_accuracy": fresh["held_out_accuracy"],
            "floor_fresh_held_out_accuracy": floor_fresh_acc,
        }

        if os.path.exists(centroids_path):
            centroids = np.load(centroids_path)

            # persisted/separation/coherence floors reuse the exact same saved
            # centroids as the trained agent -- only the encoder differs, so these
            # are a clean, controlled comparison (unlike fresh's floor, which
            # refits centroids on a different random subsample and so carries a
            # little extra noise).
            persisted = evaluate_frame_matching_persisted(
                agent, held_out_buffer, centroids, num_support=num_support, num_query=num_query,
                candidate_pool=candidate_pool, pool_size=pool_size, num_batches=num_batches)
            floor_persisted_acc = float(np.mean([
                evaluate_frame_matching_persisted(
                    fa, held_out_buffer, centroids, num_support=num_support, num_query=num_query,
                    candidate_pool=candidate_pool, pool_size=pool_size, num_batches=num_batches
                )["held_out_accuracy"] for fa in floor_agents
            ]))

            separation = evaluate_cluster_separation(agent, held_out_buffer, centroids, pool_size=pool_size, seed=seed)
            floor_separation_ratio = float(np.mean([
                evaluate_cluster_separation(fa, held_out_buffer, centroids, pool_size=pool_size, seed=seed)["separation_ratio"]
                for fa in floor_agents
            ]))

            balance = evaluate_cluster_balance(centroids, held_out_buffer, pool_size=pool_size, seed=seed)
            coherence = evaluate_cluster_state_coherence(centroids, held_out_buffer, states, pool_size=pool_size, seed=seed)

            row.update({
                "persisted_held_out_loss": persisted["held_out_loss"],
                "persisted_held_out_accuracy": persisted["held_out_accuracy"],
                "floor_persisted_held_out_accuracy": floor_persisted_acc,
                "separation_ratio": separation["separation_ratio"],
                "floor_separation_ratio": floor_separation_ratio,
                "utilization": balance["utilization"],
                "dead_clusters": balance["dead_clusters"],
                "state_coherence": coherence["overall_coherence"],
                "explained_variance": coherence["explained_variance"],
            })
        else:
            log(f"[{run}] no centroids_{checkpoint_step}.npy found -- skipping persisted-centroid checks")

        rows.append(row)

        line = (f"{run:20s} "
                f"fresh_loss={row['fresh_held_out_loss']:.3f} fresh_acc={row['fresh_held_out_accuracy']:.3f} "
                f"(floor={row['floor_fresh_held_out_accuracy']:.3f})")
        if "persisted_held_out_loss" in row:
            line += (f" persisted_loss={row['persisted_held_out_loss']:.3f} "
                     f"persisted_acc={row['persisted_held_out_accuracy']:.3f} "
                     f"(floor={row['floor_persisted_held_out_accuracy']:.3f}) "
                     f"separation={row['separation_ratio']:.2f} "
                     f"(floor={row['floor_separation_ratio']:.2f}) "
                     f"utilization={row['utilization']:.2f} "
                     f"dead={row['dead_clusters']}/{centroids.shape[0]} "
                     f"state_coherence={row['state_coherence']:.3f}")
        log(line)
        if "explained_variance" in row:
            for dim, val in row["explained_variance"].items():
                log(f"    {dim:10s} explained_variance={val:.3f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "eval_health.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(results_dir, "eval_health.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    return rows


if __name__ == "__main__":
    main()
