import os
import json
import numpy as np
import matplotlib.pyplot as plt

SOLVED_THRESHOLD = 200.0
ARM_COLORS = {"trained": "tab:blue", "random_floor": "tab:orange", "true_state": "tab:green"}


def load_run(result_dir, arm, seed):
    run_dir = os.path.join(result_dir, f"{arm}_seed{seed}")
    returns_path = os.path.join(run_dir, "episode_returns.npy")
    steps_path = os.path.join(run_dir, "episode_steps.npy")
    if not (os.path.exists(returns_path) and os.path.exists(steps_path)):
        return None, None
    return np.load(returns_path), np.load(steps_path)


def bucket_curve(returns, steps, bucket_size, num_steps):
    """Average return of every episode landing in each fixed-width step window,
    so runs with different episode lengths/counts still line up on the same x-axis."""
    num_buckets = num_steps // bucket_size
    bucket_returns = [[] for _ in range(num_buckets)]
    for r, s in zip(returns, steps):
        b = min(int(s - 1) // bucket_size, num_buckets - 1)
        bucket_returns[b].append(r)
    return np.array([np.mean(b) if b else np.nan for b in bucket_returns])


def steps_to_solve(returns, steps, window=100, threshold=SOLVED_THRESHOLD):
    """First training step at which the trailing `window`-episode average return
    reaches `threshold` -- the standard Gym "solved" definition, but tied to how
    many env steps it took to get there instead of episode count."""
    if len(returns) < window:
        return None
    rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
    hits = np.where(rolling >= threshold)[0]
    if len(hits) == 0:
        return None
    first_idx = hits[0] + window - 1  # episode that completes the first passing window
    return int(steps[first_idx])


def main(conditions=("trained", "random_floor", "true_state"), seeds=(0, 1, 2),
         result="results", bucket_size=10_000, num_steps=300_000, window=100, out_dir=None):
    out_dir = out_dir or result
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    num_buckets = num_steps // bucket_size
    bucket_step_axis = (np.arange(num_buckets) + 0.5) * bucket_size

    curves = {}       # arm -> (num_seeds_found, num_buckets) array, may contain NaN
    solve_steps = {}  # arm -> list of per-seed steps_to_solve (None if that seed never solved)

    for arm in conditions:
        per_seed_curves, per_seed_solve = [], []
        for seed in seeds:
            returns, steps = load_run(result, arm, seed)
            if returns is None:
                log(f"[{arm} seed{seed}] no episode_returns.npy/episode_steps.npy found -- skipping")
                continue
            per_seed_curves.append(bucket_curve(returns, steps, bucket_size, num_steps))
            per_seed_solve.append(steps_to_solve(returns, steps, window=window))

        if not per_seed_curves:
            continue

        curves[arm] = np.stack(per_seed_curves)
        solve_steps[arm] = per_seed_solve

        solved = [s for s in per_seed_solve if s is not None]
        if solved:
            summary = (f"mean={np.mean(solved):.0f}  min={min(solved)}  max={max(solved)}  "
                       f"({len(solved)}/{len(per_seed_solve)} seeds solved)")
        else:
            summary = f"no seed reached a {window}-episode average >= {SOLVED_THRESHOLD:.0f} within {num_steps} steps"
        log(f"{arm:14s} steps_to_solve: {summary}")

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, curve in curves.items():
        mean_curve = np.nanmean(curve, axis=0)
        std_curve = np.nanstd(curve, axis=0)
        color = ARM_COLORS.get(arm)
        ax.plot(bucket_step_axis, mean_curve, label=arm, color=color)
        ax.fill_between(bucket_step_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=color)
    ax.axhline(SOLVED_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="solved threshold")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"avg return per {bucket_size}-step window")
    ax.set_title("Learning curves (mean +/- std over seeds)")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "learning_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log(f"\nsaved plot to {plot_path}")

    with open(os.path.join(out_dir, "learning_curve.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "learning_curve.json"), "w") as f:
        json.dump({
            "bucket_size": bucket_size,
            "num_steps": num_steps,
            "window": window,
            "solved_threshold": SOLVED_THRESHOLD,
            "bucket_step_axis": bucket_step_axis.tolist(),
            "curves": {arm: curve.tolist() for arm, curve in curves.items()},
            "steps_to_solve": solve_steps,
        }, f, indent=2)

    return curves, solve_steps


if __name__ == "__main__":
    main()
