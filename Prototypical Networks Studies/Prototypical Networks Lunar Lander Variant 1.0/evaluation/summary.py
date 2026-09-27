import os
import numpy as np

from evaluation.probe import main as probing


def performance(results_dir="results", arms=("baseline", "prototype", "classifier"),
                    seeds=(0, 1, 2), window=50):
    per_run = []
    for arm in arms:
        for seed in seeds:
            path = os.path.join(results_dir, f"{arm}_seed{seed}", "episode_returns.npy")
            returns = np.load(path)
            avg_return = returns[-window:].mean()
            per_run.append({"arm": arm, "seed": seed, "avg_return": avg_return})

    per_arm = {}
    for arm in arms:
        vals = [r["avg_return"] for r in per_run if r["arm"] == arm]
        per_arm[arm] = np.mean(vals)

    return per_run, per_arm


def representation(results_dir="results", arms=("baseline", "prototype", "classifier"),
                                seeds=(0, 1, 2), checkpoint_step=500_000, num_steps=5000):
    per_run = probing(results_dir=results_dir, arms=arms, seeds=seeds,
                        checkpoint_step=checkpoint_step, num_steps=num_steps)

    per_arm = {}
    for arm in arms:
        vals = [r["probe_degradation"] for r in per_run if r["arm"] == arm]
        per_arm[arm] = np.mean(vals)

    return per_run, per_arm


def summary(results_dir="results", arms=("baseline", "prototype", "classifier"),
                    seeds=(0, 1, 2), checkpoint_step=500_000):
    _, task_per_arm = performance(results_dir, arms, seeds)
    _, robustness_per_arm = representation(results_dir, arms, seeds, checkpoint_step)

    print("Task performance (avg return over last 50 episodes, higher = better):")
    for arm in sorted(task_per_arm, key=task_per_arm.get, reverse=True):
        print(f"  {arm:12s} {task_per_arm[arm]:8.1f}")

    print("\nRepresentation robustness (probe accuracy drop under noise shift, lower = better):")
    for arm in sorted(robustness_per_arm, key=robustness_per_arm.get):
        print(f"  {arm:12s} {robustness_per_arm[arm]:.3f}")


if __name__ == "__main__":
    summary()