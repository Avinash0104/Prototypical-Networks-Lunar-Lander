# Representation Build Variant 2.3

Learns a 64-number representation of LunarLander from pixels alone, with no rewards and
no labels. The encoder is trained with a prototypical-network loss on pseudo-classes made
by clustering **frame differences**.

## How it works

1. A random policy plays `LunarLanderContinuous-v3`. Every frame is rendered to
   84x84 RGB and stored in a replay buffer (70,000 frames).
2. The input is the pixel difference between two consecutive frames (`frame[t+1] - frame[t]`),
   so it shows motion rather than the static scene.
3. K-means (16 clusters) groups those differences, using their absolute value pooled down
   to 8x8. The clusters are refit every 5,000 steps and act as the class labels.
4. Each update takes up to 16 clusters, with 4 support and 4 query samples per cluster.
   Support samples go through the encoder, and query samples go through an EMA copy of it.
   Each query is classified by its nearest cluster prototype.
5. The loss is the prototypical loss on L2-normalised embeddings.

The encoder is 4 conv layers followed by a linear layer to 64 numbers.
Situation labels (descent, approach, critical, contact) are **never used in training**.
They are only used by the evals.

## Layout

```
experiment/   environment, replay buffer, clustering, training loop, checkpoints
models/       encoder and prototypical loss
evaluation/   probe.py, health.py, visual.py
runner.py     trains 3 seeds and resumes automatically after a crash
```

## Setup

No requirements file yet. Roughly: Python 3.13, `torch`, `numpy`, `gymnasium[box2d]`,
`opencv-python`, `scikit-learn`, `matplotlib`.

## Run

Train seeds 0, 1 and 2 for 100,000 steps each, one after another, on the GPU:

```
python -u runner.py >> training_log.txt 2>&1
```

- `-u` keeps the log live. Without it, progress lines are buffered.
- If it stops, run the same command again. Each seed resumes from its latest checkpoint.
- Output goes to `results/prototype_seed{N}/`: a checkpoint and centroids every 10,000 steps,
  plus episode returns, losses and accuracies as `.npy` files.
- `runner.py` uses `device="cuda"`. Change it there to train on CPU.

## Evaluate

Run after training finishes. Each reads the step-100,000 checkpoints.

```
python -m evaluation.probe     # results/eval_probe.txt / .json
python -m evaluation.health    # results/eval_health.txt / .json
python -m evaluation.visual    # results/plots/tsne_situation.png
```

| Eval | What it reports |
|---|---|
| `probe` | Can a linear model recover the situation label and the true state (x, y, vx, vy, angle, angular velocity) from the embedding? Compared with a random untrained encoder and with a probe on the true state. |
| `health` | Held-out frame matching accuracy, cluster separation, cluster utilization and dead clusters, and how well the clusters explain the true state. |
| `visual` | t-SNE of the embeddings, coloured by situation label and by cluster. |

## Results of the final run (files since removed)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Final training accuracy | 0.979 | 0.981 | 0.978 |
| Probe accuracy | 0.857 | 0.896 | 0.899 |
| Persisted matching accuracy | 0.972 | 0.977 | 0.973 |
| State coherence | 0.458 | 0.397 | 0.416 |

- The probe floor (random encoder) was 0.790 and a linear probe on the true state scored 0.987.
- The mean probe score of 0.884 is 0.477 of the way from floor to true-state probe.
- Cluster separation was 3.5 to 4.9, against 1.7 to 2.0 for a random encoder, with no dead clusters.

## Known limitations

- Angular velocity is barely recoverable (R² near or below 0). Taking the absolute value
  of the difference before pooling erases the direction of motion.
- The `contact` situation is rare (about 2.5% of frames), and its recall in the probe is low.
