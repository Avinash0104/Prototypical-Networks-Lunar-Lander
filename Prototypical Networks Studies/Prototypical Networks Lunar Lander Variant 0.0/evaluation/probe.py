import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from evaluation.returns import load_agent
from experiment.environment import Environment


def collect_embeddings(agent, noise_mode, num_steps=5000, seed=2000):
    env = Environment(noise_mode=noise_mode, seed=seed)
    obs, info = env.reset(seed=seed)

    zs = []
    labels = []

    for _ in range(num_steps):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            z = agent.encoder(obs_t)
            action = agent.actor.deterministic(z)
            action_np = action.squeeze(0).numpy()

        zs.append(z.squeeze(0).numpy())
        labels.append(info["situation_label"])

        obs, reward, terminated, truncated, info = env.step(action_np)
        if terminated or truncated:
            obs, info = env.reset()

    return np.array(zs), np.array(labels)


def probing(agent, num_steps=5000, seed=2000):
    z_train, y_train = collect_embeddings(agent, noise_mode="walk", num_steps=num_steps, seed=seed)
    z_shift, y_shift = collect_embeddings(agent, noise_mode="iid", num_steps=num_steps, seed=seed + 1)

    z_fit, z_heldout, y_fit, y_heldout = train_test_split(z_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
    probe = LogisticRegression(max_iter=5000)
    probe.fit(z_fit, y_fit)

    train_acc = probe.score(z_heldout, y_heldout)
    shift_acc = probe.score(z_shift, y_shift)

    return train_acc, shift_acc


def main(results_dir="results", arms=("baseline", "prototype", "classifier"), seeds=(0, 1, 2), checkpoint_step=600_000, num_steps=5000):
    rows = []
    for arm in arms:
        for seed in seeds:
            run = f"{arm}_seed{seed}"
            checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
            agent = load_agent(arm, checkpoint_path)
            train_acc, shift_acc = probing(agent, num_steps=num_steps, seed=2000)

            row = {
                "arm": arm,
                "seed": seed,
                "probe_train_acc": train_acc,
                "probe_shift_acc": shift_acc,
                "probe_degradation": train_acc - shift_acc,
            }
            rows.append(row)
            print(f"{run:20s} probe_train_acc={train_acc:.3f}  probe_shift_acc={shift_acc:.3f}  "f"degradation={row['probe_degradation']:.3f}")

    return rows


if __name__ == "__main__":
    probing()