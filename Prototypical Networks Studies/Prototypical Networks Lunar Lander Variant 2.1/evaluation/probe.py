import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from evaluation.returns import load_agent
from experiment.environment import Environment, FrameStacker


def collect_embeddings(agent, num_steps=5000, seed=2000, frame_stack=4):
    env = Environment(seed=seed)
    stacker = FrameStacker(frame_stack)
    frame, info = env.reset(seed=seed)
    obs_stack = stacker.reset(frame)

    zs = []
    labels = []

    for _ in range(num_steps):
        obs_t = torch.from_numpy(obs_stack).float().unsqueeze(0)
        with torch.no_grad():
            z = agent.encoder(obs_t)
            action = agent.actor.deterministic(z)
            action_np = action.squeeze(0).numpy()

        zs.append(z.squeeze(0).numpy())
        labels.append(info["situation"])

        frame, reward, terminated, truncated, info = env.step(action_np)
        obs_stack = stacker.push(frame)
        if terminated or truncated:
            frame, info = env.reset()
            obs_stack = stacker.reset(frame)

    return np.array(zs), np.array(labels)


def probing(agent, num_steps=5000, seed=2000):
    z, y = collect_embeddings(agent, num_steps=num_steps, seed=seed)

    z_fit, z_heldout, y_fit, y_heldout = train_test_split(z, y, test_size=0.3, random_state=0, stratify=y)
    probe = LogisticRegression(max_iter=5000)
    probe.fit(z_fit, y_fit)

    return probe.score(z_heldout, y_heldout)


def main(results_dir="results", arms=("baseline", "prototype", "classifier"), seeds=(0, 1, 2), checkpoint_step=500_000, num_steps=5000):
    rows = []
    for arm in arms:
        for seed in seeds:
            run = f"{arm}_seed{seed}"
            checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
            agent = load_agent(arm, checkpoint_path)
            probe_acc = probing(agent, num_steps=num_steps, seed=2000)

            row = {
                "arm": arm,
                "seed": seed,
                "probe_acc": probe_acc,
            }
            rows.append(row)
            print(f"{run:20s} probe_acc={probe_acc:.3f}")

    return rows


if __name__ == "__main__":
    main()
