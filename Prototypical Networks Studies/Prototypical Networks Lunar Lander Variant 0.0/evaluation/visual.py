import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from evaluation.returns import load_agent
from evaluation.probe import collect_embeddings

SITUATION_LABELS = {0: "descent", 1: "approach", 2: "critical", 3: "contact"}
COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

def visualise(arm, seeds=(0, 1, 2), dir="results", checkpoint_step=600_000, num_steps=3000, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5))

    for ax, seed in zip(axes, seeds):
        run = f"{arm}_seed{seed}"
        checkpoint_path = os.path.join(dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_agent(arm, checkpoint_path)

        z, labels = collect_embeddings(agent, noise_mode="walk", num_steps=num_steps, seed=3000)
        tsne = TSNE(n_components=2, init="pca", random_state=0)
        z_2d = tsne.fit_transform(z)

        for label in sorted(set(labels)):
            mask = labels == label
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], label=SITUATION_LABELS[label], alpha=0.6, s=6, color=COLORS[label])

        ax.set_title(run)
        ax.legend(fontsize=8)

    fig.suptitle(f"t-SNE of {arm} embeddings")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"tsne_{arm}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

def main(arms=("baseline", "prototype", "classifier"), **kwargs):
    for arm in arms:
        visualise(arm, **kwargs)

if __name__ == "__main__":
    main()
