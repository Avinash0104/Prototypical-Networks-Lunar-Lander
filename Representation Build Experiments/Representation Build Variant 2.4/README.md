# Representation Build — Variant 2.4

Self-supervised representation learning on LunarLander pixels using a **prototypical-network loss over k-means pseudo-classes**, with **appearance** (stacked raw frames) as the clustering signal.

**Outcome:** the encoder learns the pretext task (it organises frames by pixel cluster) but that structure does not carry the lander's physical state. The linear probe recovers nothing beyond the majority class. See [Results](#results). This variant is worse than Variant 2.3 (motion-based clustering) on the probe; see [Comparison with 2.3](#comparison-with-variant-23).

## Method

1. **Data.** A random policy (`env.action_space.sample()`) plays `LunarLanderContinuous-v3` (`render_mode="rgb_array"`). No RL, no reward signal is used for learning. Each rendered frame is resized to 84×84 RGB.
2. **Observation.** A stack of 2 frames (6 channels). The stack never crosses an episode boundary (look-back is clamped at the episode start), and the terminal frame of each episode is stored.
3. **Pseudo-labels.** k-means (k=16) on the stack average-pooled to 16×16 (1,536 features), fit on 4,000 buffer frames and refit every 5,000 steps. Cluster membership is the class label.
4. **Episodes.** Each update draws 1,000 candidate frames from the buffer, assigns them to the nearest centroid, keeps up to 16 clusters that have at least 8 members, and takes 4 support and 4 query frames per cluster.
5. **Encoder.** 4 conv layers (32 channels, 3×3, first layer stride 2) then a linear layer to a 64-d embedding.
6. **Loss.** Prototypical loss on L2-normalised embeddings (squared Euclidean distance, no temperature). Support frames go through the online encoder and query frames through an EMA target encoder (tau 0.05).
7. **Optimiser.** Adam, lr 1e-4, one update per environment step once training starts.

The 2.4 change relative to 2.3 is step 3 plus the frame-stack input: clustering on appearance rather than on frame-to-frame differences.

## Run configuration

| | |
|---|---|
| Steps | 100,000 per seed |
| Seeds | 0, 1, 2 (run sequentially) |
| Buffer | 70,000 frames; training starts once it holds 4,000 |
| Checkpoints | every 5,000 steps |
| Device | CUDA (RTX 4050 Laptop, 6 GB) |
| Speed | about 45 ms/step in a smoke test, so roughly 75 min per seed |

The runs were interrupted several times by memory-pressure kills of the background job (seed 0 resumed from steps 55,000 and 75,000; seed 1 from 75,000) and resumed from checkpoints. Each resume refills the buffer, so about 4,000 steps run without updates after each one. `training_log.txt` has the full record.

**Environment:** Python 3.13.7, torch 2.11.0+cu128, gymnasium 1.2.3 (+ Box2D), numpy 2.5.1, scikit-learn 1.7.2, matplotlib 3.10.5, opencv 4.11.0.

## Layout

```
runner.py              supervisor: runs each seed in a worker process, restarts on any failure, resumes from checkpoints
experiment/
  training.py          Agent, train loop, resume logic
  environment.py       LunarLander wrapper, frame rendering, FrameStacker, situation labels
  buffer.py            replay buffer, frame-stack gather, cluster-based episode sampling
  clustering.py        pooled-pixel features, k-means fit, cluster assignment
  checkpoint.py        atomic checkpoint save, load, listing
models/
  encoder.py           conv encoder
  prototype.py         prototypical loss
evaluation/
  health.py            held-out task performance, cluster separation, cluster balance, state coherence
  probe.py             linear probes for situation label and continuous state
  visual.py            t-SNE plots
results/               (git-ignored) checkpoints, centroids, training logs, eval outputs, plots
```

## How to run

```
python runner.py                 # trains all seeds; safe to re-run, it skips finished seeds and resumes the rest
python -m evaluation.health      # writes results/eval_health.{txt,json}
python -m evaluation.probe       # writes results/eval_probe.{txt,json}
python -m evaluation.visual      # writes results/plots/tsne_situation.png
```

Run these from the project root. Output is also appended to `training_log.txt`. `Ctrl+C` stops the runner; running it again resumes.

## Evaluation

Each metric is compared to a **floor**: the same measurement on untrained (randomly initialised) encoders.

- **health.py**: on 5,000 held-out random-policy steps, replays the training task without gradients. "Fresh" refits centroids on the held-out data; "persisted" reuses the centroids saved at the end of training. It also reports embedding separation (mean between-cluster distance over mean within-cluster distance), cluster balance (entropy-based utilisation, dead clusters), and **state coherence**: how much of the variance in each true state dimension the pixel clusters explain.
- **probe.py**: a logistic regression predicts the 4-way situation label (descent / approach / critical / contact) and a linear regression predicts 6 state dimensions, with a train/test split by whole episodes. A probe on the true state (0.987) is the reference point for the normalised score.
- **visual.py**: t-SNE of embeddings on 3,000 held-out steps, coloured by situation label and by frame cluster.

## Results

### Health (100,000 steps, floor in parentheses)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Held-out accuracy, fresh centroids | 0.824 (0.939) | 0.848 (0.894) | 0.783 (0.883) |
| Held-out accuracy, saved centroids | 0.793 (0.897) | 0.823 (0.845) | 0.822 (0.821) |
| Separation ratio | 1.72 (1.74) | 1.81 (1.71) | 1.66 (1.61) |
| Cluster utilisation / dead clusters | 0.96 / 0 | 0.94 / 0 | 0.94 / 0 |
| State coherence (overall) | 0.188 | 0.210 | 0.134 |

State coherence by dimension (variance explained): x 0.18–0.33, vx 0.24–0.41, angle about 0.20, y 0.04–0.18, vy about 0.07, ang_vel 0.07–0.16.

Training accuracy at the end was about 0.998, so the encoder fits the training frames but does not generalise the task as well as an untrained encoder does.

### Probe

| | seed 0 | seed 1 | seed 2 | floor |
|---|---|---|---|---|
| Situation accuracy | 0.792 | 0.792 | 0.792 | 0.792 |
| Score relative to true-state probe | 0.000 | 0.000 | 0.000 | (0.987 true-state probe) |

- 0.792 is the share of "descent" frames: every seed predicts "descent" for every frame (zero precision and recall on the other three classes).
- Regression R² on the 6 state dimensions is far below zero for the trained encoders (about -6 to -890) and only about -0.5 to -1.8 for untrained ones. Rescaling the embeddings (L2-normalised or standardised) did not change this. The cause of the very negative values is not established.

### t-SNE

`results/plots/tsne_situation.png`: the embeddings form 16 tight islands, one per frame cluster, but each island mixes all four situation labels.

An ad-hoc check on seed 0 (not part of the scripts, one seed, 3,000 frames): silhouette of the cluster labels in embedding space was **0.48 trained vs 0.36 untrained**, and an untrained encoder also produces clean islands in t-SNE. Clean islands alone are therefore not evidence of learning; the silhouette gap is the modest measurable gain.

## Interpretation

- The encoder does learn the pretext structure: all 16 clusters are used, and the cluster structure is somewhat tighter than an untrained encoder's.
- That structure does not track the physics. The pixel clusters explain little of the state, and the probe finds nothing linearly recoverable.
- The pretext task saturates quickly (training accuracy above 0.99 within the first couple of thousand updates), which leaves little pressure to learn more.
- Held-out matching accuracy is below the untrained baseline, which indicates overfitting to the training buffer.

## Comparison with Variant 2.3

2.3 (sibling directory `../Representation Build Variant 2.3`) clusters and encodes the **frame difference** (motion) instead of raw frames. Shared: the loss and the training setup. Numbers below are from 2.3's saved results; the health evaluation code differs between the variants, so health numbers are only roughly comparable.

| | 2.3 | 2.4 |
|---|---|---|
| Probe accuracy, 3-seed mean | 0.884 ± 0.019 | 0.792 |
| Untrained floor | 0.790 | 0.792 |
| Score relative to true-state probe | 0.477 ± 0.097 | 0.000 |
| State coherence, seed 0 | 0.458 | 0.188 |

Motion-based clustering (2.3) gave a clearly better representation. The likely reason is that motion clusters reflect velocity and dynamics, while appearance clusters mostly reflect position and background. This has not been isolated by an experiment.

## Known limitations

- **Not exactly reproducible.** The seed controls the environment reset and k-means only. Encoder initialisation, random actions and batch sampling are unseeded, and GPU kernels are not bit-exact.
- **Resume is approximate.** The buffer and RNG state are not checkpointed, so a resumed run is not identical to an uninterrupted one. After a fallback to an older checkpoint the saved `.npy` logs can include a few entries from the rolled-back segment.
- **Loss has no temperature**, so logits are bounded and the loss cannot go much below about 1.03 even at 99.9% accuracy.
- The probe regression results (very large negative R²) are unexplained beyond the observation that they are not a scaling issue.
