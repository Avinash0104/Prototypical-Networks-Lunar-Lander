import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from experiment.training import Agent
from experiment.environment import Environment, FrameStacker

SITUATION_NAMES = ["descent", "approach", "critical", "contact"]


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
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)

    zs = []
    states = []
    labels = []

    for _ in range(num_steps):
        stack_t = torch.from_numpy(stack).float().unsqueeze(0)
        with torch.no_grad():
            z = agent.encoder(stack_t)

        zs.append(z.squeeze(0).numpy())
        states.append(info["true_state"])
        labels.append(info["situation_label"])

        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)
        if terminated or truncated:
            frame, info = env.reset()
            stack = stacker.reset(frame)

    return np.array(zs), np.array(states), np.array(labels)


def fit_probe(X, y):
    X_fit, X_heldout, y_fit, y_heldout = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    probe = LogisticRegression(max_iter=5000)
    probe.fit(X_fit, y_fit)
    y_pred = probe.predict(X_heldout)
    acc = probe.score(X_heldout, y_heldout)
    labels = list(range(len(SITUATION_NAMES)))
    report = classification_report(y_heldout, y_pred, labels=labels, target_names=SITUATION_NAMES,
                                    output_dict=True, zero_division=0)
    return acc, report


def probing(agent, num_steps=5000, seed=2000):
    z, _, y = collect_embeddings(agent, num_steps=num_steps, seed=seed)
    return fit_probe(z, y)


def baselines(num_steps=5000, seed=2000):
    z, states, y = collect_embeddings(random_agent(), num_steps=num_steps, seed=seed)
    floor_acc, _ = fit_probe(z, y)
    ceiling_acc, _ = fit_probe(states, y)
    return floor_acc, ceiling_acc


def main(results_dir="results", seeds=(0, 1, 2), checkpoint_step=100_000, num_steps=5000):
    floor_acc, ceiling_acc = baselines(num_steps=num_steps)
    print(f"{'floor (random encoder)':24s} acc={floor_acc:.3f}")
    print(f"{'ceiling (true state)':24s} acc={ceiling_acc:.3f}")

    rows = []
    for seed in seeds:
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(results_dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_agent(checkpoint_path)
        probe_acc, report = probing(agent, num_steps=num_steps, seed=2000)

        row = {"seed": seed, "probe_acc": probe_acc, "report": report}
        rows.append(row)
        print(f"{run:20s} probe_acc={probe_acc:.3f}")
        for name in SITUATION_NAMES:
            r = report[name]
            print(f"    {name:10s} precision={r['precision']:.2f} recall={r['recall']:.2f} n={int(r['support'])}")

    accs = [r["probe_acc"] for r in rows]
    avg, std = np.mean(accs), np.std(accs)
    print(f"\nRepresentation quality (avg over {len(seeds)} seeds): {avg:.3f} +/- {std:.3f}")

    return rows


if __name__ == "__main__":
    main()
