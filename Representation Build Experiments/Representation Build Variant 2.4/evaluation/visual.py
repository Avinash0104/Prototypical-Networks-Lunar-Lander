import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from evaluation.probe import load_agent, SITUATION_NAMES
from experiment.environment import Environment, FrameStacker
from experiment.clustering import frame_feature, assign_clusters, POOL_SIZE

COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

def collect_embeddings_labels_clusters(agent, centroids=None, num_steps=3000, seed=3000, pool_size=POOL_SIZE):
    env = Environment()
    env.action_space.seed(seed)
    stacker = FrameStacker()
    frame, info = env.reset(seed=seed)
    stack = stacker.reset(frame)

    zs, labels, clusters = [], [], []

    def record(stack, info):
        # the stack ends at the frame whose label `info` describes
        obs = torch.from_numpy(stack).float().unsqueeze(0)
        with torch.no_grad():
            zs.append(agent.encoder(obs).squeeze(0).numpy())
        labels.append(info["situation_label"])

        if centroids is not None:
            feat = frame_feature(obs, pool_size=pool_size).numpy()
            clusters.append(int(assign_clusters(feat, centroids)[0]))

    for _ in range(num_steps):
        record(stack, info)
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)
        stack = stacker.step(frame)
        if terminated or truncated:
            record(stack, info)  # terminal stack would otherwise be overwritten by reset()
            frame, info = env.reset()
            stack = stacker.reset(frame)

    zs = np.array(zs)
    labels = np.array(labels)
    clusters = np.array(clusters) if centroids is not None else None
    return zs, labels, clusters

def visualise(seeds=(0, 1, 2), dir="results", checkpoint_step=100_000, num_steps=3000, out_dir="results/plots", pool_size=POOL_SIZE):
    os.makedirs(out_dir, exist_ok=True)

    centroids_by_seed = {}
    for seed in seeds:
        centroids_path = os.path.join(dir, f"prototype_seed{seed}", f"centroids_{checkpoint_step}.npy")
        if os.path.exists(centroids_path):
            centroids_by_seed[seed] = np.load(centroids_path)

    show_clusters = len(centroids_by_seed) == len(seeds)
    nrows = 2 if show_clusters else 1
    fig, axes = plt.subplots(nrows, len(seeds), figsize=(6 * len(seeds), 5 * nrows), squeeze=False)

    for i, seed in enumerate(seeds):
        run = f"prototype_seed{seed}"
        checkpoint_path = os.path.join(dir, run, f"checkpoint_{checkpoint_step}.pt")
        agent = load_agent(checkpoint_path)
        centroids = centroids_by_seed.get(seed)

        z, labels, clusters = collect_embeddings_labels_clusters(
            agent, centroids=centroids, num_steps=num_steps, seed=3000, pool_size=pool_size)
        tsne = TSNE(n_components=2, init="pca", random_state=0)
        z_2d = tsne.fit_transform(z)

        ax_label = axes[0, i]
        for label in sorted(set(labels)):
            mask = labels == label
            ax_label.scatter(z_2d[mask, 0], z_2d[mask, 1], label=SITUATION_NAMES[label], alpha=0.6, s=6, color=COLORS[label])
        ax_label.set_title(f"{run}\ncolored by situation label")
        ax_label.legend(fontsize=8)

        if show_clusters:
            ax_cluster = axes[1, i]
            scatter = ax_cluster.scatter(z_2d[:, 0], z_2d[:, 1], c=clusters, cmap="nipy_spectral",
                                          vmin=0, vmax=centroids.shape[0] - 1, alpha=0.6, s=6)
            ax_cluster.set_title(f"{run}\ncolored by frame cluster")
            fig.colorbar(scatter, ax=ax_cluster, label="cluster id", fraction=0.046, pad=0.04)

    fig.suptitle("t-SNE of embeddings")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "tsne_situation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

if __name__ == "__main__":
    visualise()
