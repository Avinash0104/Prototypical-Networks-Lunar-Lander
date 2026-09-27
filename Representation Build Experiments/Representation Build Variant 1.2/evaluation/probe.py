import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, r2_score

from experiment.training import Agent
from experiment.environment import Environment, FrameStacker

SITUATION_NAMES = ["descent", "approach", "critical", "contact"]
STATE_DIM_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def load_agent(checkpoint_path, device="cpu"):
    agent = Agent(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.encoder.eval()
    return agent


def random_agent(device="cpu"):
    agent = Agent(device=device)
    agent.encoder.eval()
    return agent


def collect_embeddings(agent, num_steps=5000, seed=2000):
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)

    zs = []
    states = []
    labels = []
    episode_ids = []
    episode_id = 0

    for _ in range(num_steps):
        stack_t = torch.from_numpy(stack).float().unsqueeze(0)
        with torch.no_grad():
            z = agent.encoder(stack_t)

        zs.append(z.squeeze(0).numpy())
        states.append(info["true_state"])
        labels.append(info["situation_label"])
        episode_ids.append(episode_id)

        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)
        if terminated or truncated:
            frame, info = env.reset()
            stack = stacker.reset(frame)
            episode_id += 1

    return np.array(zs), np.array(states), np.array(labels), np.array(episode_ids)


def episode_train_test_split(episode_ids, test_size=0.3, seed=0):
    # Splitting whole episodes (rather than random frames) keeps near-duplicate
    # adjacent frames from a continuous rollout on the same side of the split --
    # a random per-frame split leaks, since consecutive frames barely differ.
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_episodes)

    target_test_count = test_size * len(episode_ids)
    test_episodes = []
    running_count = 0
    for ep in shuffled:
        if running_count >= target_test_count:
            break
        test_episodes.append(ep)
        running_count += np.sum(episode_ids == ep)

    test_mask = np.isin(episode_ids, test_episodes)
    train_mask = ~test_mask
    return train_mask, test_mask


def fit_probe(X, y, episode_ids):
    train_mask, test_mask = episode_train_test_split(episode_ids, test_size=0.3, seed=0)
    X_fit, X_heldout = X[train_mask], X[test_mask]
    y_fit, y_heldout = y[train_mask], y[test_mask]

    probe = LogisticRegression(max_iter=5000)
    probe.fit(X_fit, y_fit)
    y_pred = probe.predict(X_heldout)
    acc = probe.score(X_heldout, y_heldout)
    labels = list(range(len(SITUATION_NAMES)))
    report = classification_report(y_heldout, y_pred, labels=labels, target_names=SITUATION_NAMES,
                                    output_dict=True, zero_division=0)
    return acc, report


def fit_regression_probe(X, states, episode_ids):
    # Regressing z onto the raw continuous state (rather than only classifying
    # the hand-picked situation_label buckets) shows how much real physical
    # state is linearly recoverable, without presupposing any one discretization.
    Y = states[:, :len(STATE_DIM_NAMES)]
    train_mask, test_mask = episode_train_test_split(episode_ids, test_size=0.3, seed=0)
    X_fit, X_heldout = X[train_mask], X[test_mask]
    Y_fit, Y_heldout = Y[train_mask], Y[test_mask]

    reg = LinearRegression()
    reg.fit(X_fit, Y_fit)
    Y_pred = reg.predict(X_heldout)
    r2_per_dim = r2_score(Y_heldout, Y_pred, multioutput="raw_values")
    return dict(zip(STATE_DIM_NAMES, (float(r2) for r2 in r2_per_dim)))


def probing(agent, num_steps=5000, seed=2000):
    z, states, y, episode_ids = collect_embeddings(agent, num_steps=num_steps, seed=seed)
    acc, report = fit_probe(z, y, episode_ids)
    r2_per_dim = fit_regression_probe(z, states, episode_ids)
    return acc, report, r2_per_dim


def baselines(num_steps=5000, seed=2000, num_floor_draws=3):
    # floor is architecture-dependent (random-projection statistics of an untrained
    # encoder), so it's averaged over several fresh random inits rather than one
    # noisy draw; ceiling is fit on the true physics state, independent of the
    # encoder entirely, so one fit is enough
    floor_accs = []
    floor_r2s = []
    ceiling_acc = None
    for _ in range(num_floor_draws):
        z, states, y, episode_ids = collect_embeddings(random_agent(), num_steps=num_steps, seed=seed)
        floor_acc, _ = fit_probe(z, y, episode_ids)
        floor_accs.append(floor_acc)
        floor_r2s.append(fit_regression_probe(z, states, episode_ids))
        if ceiling_acc is None:
            ceiling_acc, _ = fit_probe(states, y, episode_ids)

    avg_floor_r2 = {dim: float(np.mean([r[dim] for r in floor_r2s])) for dim in STATE_DIM_NAMES}
    return float(np.mean(floor_accs)), ceiling_acc, avg_floor_r2


def normalized_score(probe_acc, floor_acc, ceiling_acc, eps=1e-6):
    """Fraction of the floor-to-ceiling gap that training closed. Lets probe_acc be
    compared across architectures whose untrained floor differs (see baselines)."""
    gap = ceiling_acc - floor_acc
    if gap < eps:
        raise ValueError(f"floor/ceiling gap too small to normalize ({gap:.4f}) — eval setup is likely broken")
    return (probe_acc - floor_acc) / gap


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=5000):
    floor_acc, ceiling_acc, floor_r2 = baselines(num_steps=num_steps)
    print(f"{'floor (random encoder)':24s} acc={floor_acc:.3f}")
    print(f"{'ceiling (true state)':24s} acc={ceiling_acc:.3f}")
    print("floor R^2 (random encoder, continuous state):")
    for dim in STATE_DIM_NAMES:
        print(f"    {dim:10s} R^2={floor_r2[dim]:.3f}")

    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_agent(checkpoint_path)
        probe_acc, report, r2_per_dim = probing(agent, num_steps=num_steps, seed=2000)
        norm_score = normalized_score(probe_acc, floor_acc, ceiling_acc)

        row = {"seed": seed, "probe_acc": probe_acc, "normalized_score": norm_score,
               "report": report, "r2_per_dim": r2_per_dim}
        rows.append(row)
        print(f"{run:20s} probe_acc={probe_acc:.3f}  normalized={norm_score:.3f}")
        for name in SITUATION_NAMES:
            r = report[name]
            print(f"    {name:10s} precision={r['precision']:.2f} recall={r['recall']:.2f} n={int(r['support'])}")
        print("    continuous-state R^2 (vs floor):")
        for dim in STATE_DIM_NAMES:
            print(f"        {dim:10s} R^2={r2_per_dim[dim]:.3f}  (floor={floor_r2[dim]:.3f})")

    accs = [r["probe_acc"] for r in rows]
    norms = [r["normalized_score"] for r in rows]
    avg, std = np.mean(accs), np.std(accs)
    norm_avg, norm_std = np.mean(norms), np.std(norms)
    print(f"\nRepresentation quality (avg over {len(seeds)} seeds): {avg:.3f} +/- {std:.3f}")
    print(f"Normalized score, floor=0/ceiling=1 (avg over {len(seeds)} seeds): {norm_avg:.3f} +/- {norm_std:.3f}")

    for dim in STATE_DIM_NAMES:
        dim_r2s = [r["r2_per_dim"][dim] for r in rows]
        print(f"{dim:10s} R^2 avg={np.mean(dim_r2s):.3f} +/- {np.std(dim_r2s):.3f}  (floor={floor_r2[dim]:.3f})")

    return rows


if __name__ == "__main__":
    main()
