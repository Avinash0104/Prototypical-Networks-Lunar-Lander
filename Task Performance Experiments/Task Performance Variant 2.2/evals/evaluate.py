import os
import json
import torch
import numpy as np
import gymnasium as gym

from experiment.environment import Environment
from experiment.checkpoint import load_pretrained_encoder, find_latest_checkpoint, load_checkpoint
from experiment.training import Agent
from models.encoder import Encoder

SOLVED_THRESHOLD = 200.0


def build_env(seed, arm, pretrained_step=100_000, device="cpu"):
    if arm == "true_state":
        env = gym.make("LunarLanderContinuous-v3")
        obs_dim = env.observation_space.shape[0]
        return env, obs_dim

    torch.manual_seed(seed)
    encoder = Encoder().to(device)
    if arm == "trained":
        pretrained_path = os.path.join("checkpoints", f"prototype_seed{seed}", f"checkpoint_{pretrained_step}.pt")
        load_pretrained_encoder(encoder, pretrained_path, device=device)
    env = Environment(seed=seed, encoder=encoder)
    obs_dim = env.observation_space.shape[0]
    return env, obs_dim


def evaluate(arm, seed, result="results", pretrained_step=100_000, num_episodes=100, device="cpu"):
    run_name = f"{arm}_seed{seed}"
    dir = os.path.join(result, run_name)

    checkpoint_path, trained_step = find_latest_checkpoint(dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"no checkpoint found in {dir}")

    env, obs_dim = build_env(seed, arm, pretrained_step=pretrained_step, device=device)
    agent = Agent(obs_dim=obs_dim, device=device)
    load_checkpoint(agent, checkpoint_path)

    returns = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=10_000 + ep)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        returns.append(episode_return)

    returns = np.array(returns)
    return {
        "arm": arm,
        "seed": seed,
        "trained_step": trained_step,
        "num_episodes": num_episodes,
        "avg_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "success_rate": float((returns >= SOLVED_THRESHOLD).mean()),
        "returns": returns.tolist(),
    }


def main(conditions=("trained", "random_floor", "true_state"), seeds=(0, 1, 2), result="results", pretrained_step=100_000, num_episodes=100):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    rows = []
    for arm in conditions:
        for seed in seeds:
            row = evaluate(arm, seed, result=result, pretrained_step=pretrained_step, num_episodes=num_episodes)
            rows.append(row)
            log(f"{arm:14s} seed{seed}  trained_to_step={row['trained_step']:>7d}  "
                f"avg_return={row['avg_return']:.1f} +/- {row['std_return']:.1f}  "
                f"success_rate={row['success_rate']:.2f}")

    log()
    aggregates = {}
    for arm in conditions:
        arm_rows = [r for r in rows if r["arm"] == arm]
        returns_by_seed = [r["avg_return"] for r in arm_rows]
        succ_by_seed = [r["success_rate"] for r in arm_rows]

        avg, avg_std = float(np.mean(returns_by_seed)), float(np.std(returns_by_seed))
        succ, succ_std = float(np.mean(succ_by_seed)), float(np.std(succ_by_seed))
        aggregates[arm] = {
            "avg_return_mean": avg, "avg_return_std": avg_std,
            "avg_return_min": float(min(returns_by_seed)), "avg_return_max": float(max(returns_by_seed)),
            "success_rate_mean": succ, "success_rate_std": succ_std,
            "success_rate_min": float(min(succ_by_seed)), "success_rate_max": float(max(succ_by_seed)),
        }
        log(f"{arm:14s} avg_return={avg:.1f}+/-{avg_std:.1f} (seed range {min(returns_by_seed):.1f} to {max(returns_by_seed):.1f})  "
            f"success_rate={succ:.2f}+/-{succ_std:.2f} (seed range {min(succ_by_seed):.2f} to {max(succ_by_seed):.2f})")

    # With only a handful of seeds, a formal significance test is more misleading than
    # useful -- instead check whether the seed ranges even overlap, a cheaper and more
    # honest signal at this sample size.
    log()
    if "trained" in aggregates and "random_floor" in aggregates:
        trained_returns = [r["avg_return"] for r in rows if r["arm"] == "trained"]
        floor_returns = [r["avg_return"] for r in rows if r["arm"] == "random_floor"]
        non_overlapping = min(trained_returns) > max(floor_returns)
        log(f"trained's worst seed ({min(trained_returns):.1f}) "
            f"{'beats' if non_overlapping else 'does NOT clearly beat'} "
            f"random_floor's best seed ({max(floor_returns):.1f}) -- "
            + ("a robust signal given only 3 seeds per arm."
               if non_overlapping else
               "seed ranges overlap, so treat the average difference cautiously with only 3 seeds per arm."))

    os.makedirs(result, exist_ok=True)
    with open(os.path.join(result, "eval_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(result, "eval_results.json"), "w") as f:
        json.dump({"rows": rows, "aggregates": aggregates}, f, indent=2)

    return rows


if __name__ == "__main__":
    main()