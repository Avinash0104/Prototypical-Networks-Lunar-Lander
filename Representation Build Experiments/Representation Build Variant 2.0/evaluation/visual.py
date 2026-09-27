import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from evaluation.probe import collect_embeddings, load_agent, SITUATION_NAMES

COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

def visualise(seeds=(0, 1, 2), dir="results", checkpoint_step=100_000, num_steps=3000, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 5), squeeze=False)

    for i, seed in enumerate(seeds):
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_agent(checkpoint_path)

        z, _, labels = collect_embeddings(agent, num_steps=num_steps, seed=3000)
        tsne = TSNE(n_components=2, init="pca", random_state=0)
        z_2d = tsne.fit_transform(z)

        ax_label = axes[0, i]
        for label in sorted(set(labels)):
            mask = labels == label
            ax_label.scatter(z_2d[mask, 0], z_2d[mask, 1], label=SITUATION_NAMES[label], alpha=0.6, s=6, color=COLORS[label])
        ax_label.set_title(f"{run}\ncolored by situation label")
        ax_label.legend(fontsize=8)

    fig.suptitle("t-SNE of embeddings")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "tsne_situation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

if __name__ == "__main__":
    visualise()
