# Prototypical Networks in Reinforcement Learning State Spaces

Sep 26, 2026 · Avinash Lakhwani

## Preface

This project tested whether prototypical-network objectives give a reinforcement-learning agent better state representations on `LunarLanderContinuous-v3`. Work falls into three study lines, all kept as versioned folders in the project directory.

**Study line 1 (auxiliary losses).** A Soft Actor-Critic agent was given a supervised prototype loss on its encoder, over four hand-defined "situation" labels. It was compared with a plain agent and with an ordinary classifier head, first on vector observations padded with noise distractors, then on pixels. **Study line 2 (representation building)** pre-trained pixel encoders without reward, using self-supervised prototypical objectives. **Study line 3 (task performance)** froze the best of those encoders and trained Soft Actor-Critic on top.

Adding a prototype loss to the agent's training gave at most a small, inconsistent gain in robustness to noise, and it lowered task performance. Training the encoder separately, without reward, worked better as representation learning. Clustering frame differences and training a prototypical loss on those clusters produced an encoder whose categories generalized to new data and tracked the lander's position and speed, though not its rotation. When that encoder was frozen and used as the input to a Soft Actor-Critic agent, the agent performed worse than one given an untrained encoder. A well-organized representation was not enough on its own for control.

## 1. Introduction

Reinforcement learning agents that learn from raw or high-dimensional observations must work out which parts of the input matter for control. Irrelevant parts, such as background, lighting or sensor noise, can be picked up as spurious features. When those parts change at test time, the policy can fail even though the task is unchanged. Auxiliary and self-supervised representation objectives are a common remedy.

Prototypical networks \[1\] classify an input by its distance to a learned "prototype" for each class, usually the mean embedding of a few labelled examples. The geometry is simple: each class is a point, and a good embedding pulls same-class inputs together and pushes classes apart. Yarats et al. \[2\] carried this idea into reinforcement learning. It learns a set of prototypes over observation embeddings without labels and uses them both to shape the encoder and to drive exploration.

The project started from one question, set out in the original project specification for Study 0.0:

> Does adding a supervised prototype loss to a reinforcement learning agent's encoder make its state representations more robust to irrelevant or noisy observation dimensions, compared to a plain agent and an agent with an ordinary auxiliary classifier head?

As the work went on, two further questions were added:

1. Can a prototypical objective, trained **without reward or labels**, build a pixel representation that carries the lander's physical state?
2. Does such a pre-trained representation help a Soft Actor-Critic agent **perform the task** when it is used as a frozen input?

The project's contributions are:

- A controlled three-arm comparison (baseline, prototype, classifier) that differs only in the auxiliary loss, on both a vector-with-distractors and a pixel version of Lunar Lander.
- Eight variants of self-supervised pixel representation learning, covering Yarats-style Sinkhorn prototypes, instance-level prototypical loss and k-means pseudo-class episodes. Each is scored against untrained-encoder floors with linear probes and cluster-health metrics.
- A downstream test that separates the representation from control by freezing the encoder and comparing it with a random encoder and with the true state.
- Written diagnoses of failure modes in the code base, including embedding-scale inflation, sampling duplicates, label timing and unseeded evaluation.

## 2. Background and related work

### 2.1 Prototypical networks

Snell et al. \[1\] train an embedding fφ on few-shot "episodes". Each episode has a support set S\_k and a query set for each of N classes. The prototype of class k is the mean support embedding, and a query x is classified by a softmax over negative squared Euclidean distances:

- **Prototype of class k:** c(k) = the average of f(x) over the support examples of class k
- **Probability that x belongs to class k:** p(k | x) = exp(−‖f(x) − c(k)‖²) ÷ \[ sum over every class j of exp(−‖f(x) − c(j)‖²) \]

The loss is the negative log-probability of the true class. The project keeps a PyTorch re-implementation of this method (the orobix repository \[6\]), and the project's own prototypical-loss code follows it closely.

### 2.2 Prototypes in reinforcement learning

The method of Yarats et al. \[2\] pre-trains a convolutional encoder on reward-free interaction. It keeps a bank of learnable unit-norm prototypes. An online encoder with a predictor must match the prototype assignment that a slow-moving target encoder, updated as an exponential moving average, gives the next observation. Assignments are balanced across prototypes with a few Sinkhorn–Knopp iterations \[5\], as in Swapping Assignments between Views \[4\]. The prototypes also supply a k-nearest-neighbor entropy bonus for exploration. The project keeps the official code \[7\], which is the template for Study 3.0 and Representation Build 1.x.

### 2.3 Auxiliary and self-supervised objectives for reinforcement learning

Auxiliary losses that share the policy's encoder can speed learning and shape features. Contrastive objectives such as Contrastive Unsupervised Representations for Reinforcement Learning and augmentation methods such as Data-regularized Q-learning made pixel-based Soft Actor-Critic competitive with state-based Soft Actor-Critic on standard benchmarks. The exponential-moving-average target networks used here for query embeddings come from bootstrap methods such as Bootstrap Your Own Latent. Work on visual distractors shows that task-irrelevant variation can sharply cut returns, which is why Study 0.0 uses a distractor wrapper.

### 2.4 Soft Actor-Critic

Soft Actor-Critic \[3\] is an off-policy actor-critic method that maximizes return plus policy entropy. It uses twin action-value networks, a tanh-squashed Gaussian policy and an automatically tuned temperature α. Every agent in the project uses a from-scratch Soft Actor-Critic with these components.

### 2.5 The environment

`LunarLanderContinuous-v3` in Gymnasium has an 8-dimensional state: horizontal and vertical position, horizontal and vertical velocity, angle, angular velocity and two leg-contact flags. The action is a 2-dimensional continuous throttle. An episode return of 200 or more counts as solved. The project renders the environment to 84×84 frames for its pixel variants.

## 3. Project structure

The project has three experiment families. Each family is a folder of numbered, self-contained variants, and each variant is organised the same way. There is code for the experiment itself (environment wrapper, replay buffer, training loop, checkpointing), for the models (encoder, Soft Actor-Critic heads, prototype or classifier modules) and for evaluation (returns, linear probe, health metrics, t-distributed stochastic neighbour embedding). Checkpoints, return curves and evaluation reports sit next to the code. The families build on each other. The studies put prototypes inside reinforcement learning training. The Representation Build experiments train encoders without reward. The Task Performance experiments test those encoders on the control task.

### 3.1 Prototypical Networks Studies

**Purpose:** test whether adding a prototype loss to the Soft Actor-Critic agent's encoder, while the agent learns the task, makes its state representation more robust. Robustness here means the representation keeps working when irrelevant parts of the observation change, such as added noise dimensions. The prototype agent is compared with a plain agent and with an agent given an ordinary classifier head trained on the same labels. That way, any gain can be credited to the prototype geometry rather than simply to extra supervision. The studies started on vector observations, where noise could be controlled exactly, and then moved to pixels to see whether the idea carries over to images.

| Variant | Observation | Focus | Status |
| --- | --- | --- | --- |
| 0.0 | 8-dimensional state + noise dimensions | Three arms (baseline, prototype bank, classifier) under distractor noise | 3 result sets × 3 arms × 3 seeds, 600,000 steps |
| 1.0 | 4 stacked 84×84 frames | Prototype-bank auxiliary loss moved to pixels | 1-seed runs, 150,000–300,000 steps |
| 2.0 / 2.1 | Vector, then pixels | Episodic Snell-style prototype loss with norm penalty and gradient balancing; 2.1 adds random crop | 3-seed vector run; 1-seed pixel run to 500,000 steps |
| 3.0 | Pixels | Yarats-style reward-free pre-training with exploration bonus | Code only |

### 3.2 Representation Build Experiments

**Purpose:** find out whether a prototypical objective can build a useful pixel representation on its own, before any control is learned. The encoder is trained only on frames from a random policy, with no reward and no labels, so whatever it learns must come from the structure of the images themselves. Separating this step from control removes the conflict seen in the studies, where the prototype loss competed with the agent's value learning. Each variant is measured on how much of the lander's physical state (position, velocity, angle and rotation speed) its clusters capture, and on how well that structure holds up on frames it has not seen. The variants differ in the prototypical objective and in the signal used to group frames: raw appearance, or motion between frames.

| Variant | Objective | Input |
| --- | --- | --- |
| 1.0 / 1.1 | Yarats-style Sinkhorn prototypes (32) | 3 stacked colour frames |
| 1.2 | Yarats-style Sinkhorn prototypes (32) | Frame differences |
| 2.0 | Instance-level prototypical loss | 3 stacked colour frames |
| 2.1–2.3 | Prototypical loss over k-means motion clusters (2.3 adds unit-length normalisation) | Frame difference |
| 2.4 | Prototypical loss over k-means appearance clusters | 2 stacked colour frames |

### 3.3 Task Performance Experiments

**Purpose:** find out whether those representations actually help an agent learn to land. Good cluster structure is only useful if a policy can act on it. So each pre-trained encoder is given to a Soft Actor-Critic agent as its only view of the world, and the agent learns from reward. It is compared with the same agent using an untrained encoder of the same design, which shows whether pre-training added anything beyond random features. It is also compared with an agent given the true 8-dimensional state, which shows the best this agent can do at this budget. The variants change how the encoder is used, first frozen and then fine-tuned during learning, to test whether reward can fill in what pre-training missed.

| Variant | Encoder tested | How the encoder is used | Conditions compared | Change from the previous variant |
| --- | --- | --- | --- | --- |
| 2.2 | Representation Build 2.2 (motion clusters, unnormalised) | Frozen; Soft Actor-Critic sees the 64-dimensional embedding of each frame difference | trained, random\_floor, true\_state; 3 seeds × 200,000 steps | First downstream test |
| 2.3 | Representation Build 2.3 (motion clusters, unit-length) | Frozen, as in 2.2 | Same | Actor gets a 128-unit hidden layer; torch seeded per run |
| 2.4 | Pre-trained motion encoder (frame-difference input) | Fine-tuned: critic gradients update the encoder, with gradient clipping at norm 10 | trained versus random-initialised encoder | Unfreezes the encoder; buffer stores each frame once. Still running, no results yet |

## 4. Methodology

The project's method has three parts:

1. The prototypical network studies added a prototype loss directly to Soft Actor-Critic's encoder during training (4.1).
2. That did not work reliably, so a pixel encoder was then trained on its own with a prototypical objective, using no reward and no labels (4.2).
3. That encoder was given to a Soft Actor-Critic agent to test whether it helps control in an RL environment(4.3).

Every experiment maps the observation to a 64-dimensional embedding. The prototypical losses either follow Snell-style episodes, where each prototype is the mean embedding of its support examples, or use a learnable bank of unit-norm prototypes with Sinkhorn-balanced assignments.

### 4.1 Prototypical Networks

Each study compares three agents that differ only in the auxiliary term added to the critic loss. The shared encoder is updated by critic and auxiliary gradients, and actor gradients are stopped:

- **Total loss** = critic loss + λ(t) × auxiliary loss
- **λ(t)** = 0 for the first 10,000 steps, rises linearly from 0 to 1 between 10,000 and 20,000 steps, and stays at 1 after that

* **Baseline:** no auxiliary loss.
* **Classifier control:** a two-layer multilayer perceptron on z, trained with cross-entropy.
* **Prototype:** a prototype loss on a projection of z (forms below).

The supervised losses use four rule-based "situation" classes computed from the true state: contact (a leg touching), critical (y < 0.3 with speed > 0.6 or tilt > 0.4), high descent (y > 0.6) and approach (everything else). The classes are very unbalanced: about 79% descent and 2.5% contact under a random policy.

The vector studies add a distractor wrapper. It appends noise dimensions that are independent of the state: 32 in the specification, 128 in the final code. During training the noise is a clipped random walk (step size 0.02, clipped to ±3). The robustness test replaces the walk with independent standard-normal noise drawn fresh at every step. A linear probe is fitted on training-noise embeddings and tested on shift-noise embeddings. A smaller accuracy drop means a more noise-invariant representation.

### 4.2 Representation building (reward-free)

A random policy plays the environment and fills a 70,000-frame buffer. A small convnet (four 3×3 conv layers of 32 channels, then a linear layer to 64-dimensional) is trained for 100,000 steps. No reward and no situation labels are used. Three families of objective were tried:

1. **Yarats style (1.0–1.2).** There are 32 learnable prototypes. The online encoder plus predictor predicts the Sinkhorn-balanced prototype assignment (3 iterations) that an exponential-moving-average target encoder gives the next frame stack. Random-shift augmentation is applied. Variant 1.2 feeds the encoder frame differences instead of raw frames.
2. **Instance-level prototypical loss (2.0).** Each frame in a batch is its own class. K augmented copies form the support, and the target encoder's embedding of the next frame is the query.
3. **Pseudo-class episodes (2.1–2.4).** k-means (k = 16) is fitted to pooled pixel features and refitted every 5,000 steps. Cluster numbers act as class labels for Snell-style episodes with 4 support and 4 query frames per cluster. Support frames go through the online encoder and query frames through an exponential-moving-average target (τ = 0.05). 2.2 and 2.3 cluster the absolute frame difference, pooled to 8×8, so they capture motion. 2.4 clusters two stacked raw frames pooled to 16×16, so it captures appearance. 2.3 scales embeddings and prototypes to unit length, a fix proposed in the 2.2 review so that the loss cannot be lowered just by making embeddings larger.

### 4.3 Task performance with pre-trained encoders

The Task Performance experiments ask whether a representation learned without reward is useful to an agent that must learn from reward. They isolate the representation by holding everything else fixed: the same Soft Actor-Critic agent, environment, step budget and seeds. Only the source of the agent's observation changes.

**How an observation is produced.** At every environment step:

1. The encoder maps the frames or difference in frames to a 64-dimensional embedding. In 2.2 and 2.3 no gradient reaches the encoder.
2. Soft Actor-Critic receives only this embedding as its state; it never sees the pixels or the true physics. Its actor and twin critics are small multilayer perceptrons on the embedding (§6.1).
3. Soft Actor-Critic then learns from the standard Lunar Lander reward for 200,000 steps. The first 1,000 steps use random actions, and embeddings are stored in a 200,000-transition replay buffer.

**Conditions compared.** Each condition has a specific job:

- **trained:** the encoder is the step-100,000 Representation Build checkpoint for the matching seed. This condition measures what pre-training delivers.
- **random\_floor:** the same architecture, randomly initialised and frozen. This controls for the architecture. An untrained convnet still passes on a lot of information as random features, so `trained` must beat it before pre-training can be said to add anything.
- **true\_state:** Soft Actor-Critic on the raw 8-dimensional state, with no encoder. This is the ceiling: what this agent can reach at this budget with perfect information.

Comparing `trained` with `random_floor` gives the value of pre-training. Comparing `trained` with `true_state` shows how much control-relevant information the representation still lacks.

**Variants.**

- **2.2** uses the Representation Build 2.2 encoders.
- **2.3** uses the 2.3 encoders, adds a 128-unit hidden layer to the actor, and seeds PyTorch per run.
- **2.4** unfreezes the encoder: critic gradients fine-tune it, clipped to norm 10. This tests whether reward can repair features the pre-training missed.

After training, each agent is evaluated on 100 deterministic episodes (§6.2).

## 5. Prototypical network studies

The four studies (0.0 to 3.0) tested whether a prototype loss added to Soft Actor-Critic's encoder during training makes its representation more robust. It did not do so reliably, and it usually lowered return. That result led the project to separate representation learning from control (Sections 4, 6 and 7).

### 5.1 Progress across variants

| Variant | What changed | Outcome |
| --- | --- | --- |
| 0.0 | Prototype bank: 4 unit-norm prototypes, cosine logits with τ = 0.1. The final code adds a noise-consistency term and prototype repulsion. Three evaluation passes, 600,000 steps each | No consistent robustness gain; prototype arm had the lowest or near-lowest returns |
| 1.0 | Same loss on 4 stacked 84×84 pixel frames | No arm learned to land (last-50 mean −462 to −606) |
| 2.0 / 2.1 (vector) | Episodic Snell loss (20 support, 64 query per class), embedding-size penalty, exponential-moving-average gradient-norm balancing | Best probe robustness of any arm (0.296), but worst return (−166) |
| 2.0 / 2.1 (pixels) | Same, plus ±4 px random crop in 2.1 | No arm learned to land (−279 to −388) |
| 3.0 | Yarats-style reward-free pre-training with predictor, exponential-moving-average target and exploration bonus | Code only; became the starting point for Representation Build 1.x |

### 5.2 Results

**Table 1. Probe degradation and returns in the vector studies.** Degradation is in-distribution probe accuracy minus shift-noise accuracy (lower is better). Returns are means over 30 deterministic episodes. Values are 3-seed means ± standard deviation.

| Run | Arm | Probe degradation | Return, train noise | Return, shift noise |
| --- | --- | --- | --- | --- |
| 0.0, first pass | Baseline | **0.300 ± 0.136** | **−22.6 ± 41.4** | 31.6 |
|  | Prototype | 0.507 ± 0.084 | −106.9 ± 11.9 | −30.7 |
|  | Classifier | 0.372 ± 0.066 | −69.9 ± 29.7 | −51.1 |
| 0.0, third pass | Baseline | 0.473 ± 0.008 | −98.5 ± 52.1 | −29.4 |
|  | Prototype | 0.348 ± 0.060 | **−92.8 ± 23.5** | 27.8 |
|  | Classifier | **0.234 ± 0.211** | −114.9 ± 110.9 | −59.6 |
| 2.x vector, episodic loss | Baseline | 0.333 | −98.5 ± 31.6 | −15.8 |
|  | Prototype | **0.296** | −166.2 ± 62.4 | −56.4 |
|  | Classifier | 0.361 | **−94.3 ± 23.1** | 38.9 |

The arms' rankings flip from one run to the next. In-distribution probe accuracy was 0.95–1.00 for every arm, so all encoders carried the labels before the shift. In t-distributed stochastic neighbour embedding plots every arm forms episode-shaped curves rather than label clusters.

### 5.3 What the studies showed

- **The effect is within seed noise.** With 3 seeds, standard deviations (0.06–0.21) are as large as the gaps between arms, so no robustness claim holds in either direction.
- **The loss competes with the critic.** A code review during the studies named four causes: unbounded embedding scale, class-balanced episodes pulling against imbalanced critic batches, rare classes resampled with replacement, and λ = 1 with no gradient balance. Fixing the first and last in 2.x did not bring returns back.
- **The labels are too easy.** Even the baseline reached near-perfect probe accuracy, so the supervised signal added little.
- **The shift test did not stress the policy.** Returns rose under independent per-step noise in most rows, so the return comparison says little about fragility.
- **Pixel Soft Actor-Critic did not learn at these budgets** for any arm, so pixel comparisons between arms tell us nothing.

These results pointed the project toward training the representation separately, without reward, and then testing it on control.

## 6. Experimental setup

This setup covers the Representation Build and Task Performance experiments. Both use `LunarLanderContinuous-v3` rendered to 84×84 colour frames. The Representation Build runs used 3 seeds each, and a supervisor script resumed them from checkpoints after crashes.

### 6.1 Architecture

Every learned representation comes from one small convolutional encoder. The Representation Build variants add training heads to it, and the Task Performance variants put a Soft Actor-Critic agent on top of it. Components and their settings:

| Component | Used in | Structure |
| --- | --- | --- |
| Pixel encoder | All Representation Build variants; Task Performance `trained` and `random_floor` | 84×84 input scaled to \[0, 1\] → Convolution 3×3, 32 channels, stride 2 → 3 × (Convolution 3×3, 32 channels, stride 1), rectified linear unit after each → flatten → Linear → 64-dimensional z |
| Encoder input | By variant | 3 stacked colour frames (9 channels) in 1.0, 1.1 and 2.0; 2 frame differences (6 channels) in 1.2; one signed frame difference (3 channels) in 2.1–2.3; 2 stacked frames (6 channels) in 2.4 |
| exponential-moving-average target encoder | Representation Build | Copy of the encoder, Polyak-averaged with τ = 0.05, no gradient; embeds next frames (1.x) or query frames (2.x) |
| Predictor | Representation Build 1.x | Multilayer perceptron 64 → 128 → 64 on the online embedding |
| Prototype bank | Representation Build 1.x | 32 learnable unit-norm vectors in R⁶⁴; logits = cosine similarity / 0.1; targets from 3 Sinkhorn iterations; renormalised after every step |
| Prototypical head | Representation Build 2.x | No parameters: prototype = mean support embedding; logits = −squared Euclidean distance (2.3 scales embeddings and prototypes to unit length first) |
| Pseudo-labeller | Representation Build 2.1–2.4 | k-means, k = 16, on the absolute frame difference average-pooled to 8×8 (2.1–2.3) or the 2-frame stack pooled to 16×16 (2.4); refitted every 5,000 steps on 4,000 buffer frames |
| Soft Actor-Critic actor | Task Performance | Tanh-squashed Gaussian, log standard deviation clamped to \[−20, 2\]. 2.2: linear mean and log standard deviation heads on z. 2.3 and 2.4: a 128-unit rectified linear unit layer before the heads |
| Soft Actor-Critic critics | Task Performance | Twin action-value networks, (z, a) → 128 rectified linear unit → 1; target critics Polyak-averaged with τ = 0.005 |
| Temperature | Task Performance | Learned log α, target entropy −2 |

In Task Performance 2.2 and 2.3 the encoder is frozen (2.4 fine-tunes it): the environment wrapper runs it on each transition's frame difference, and Soft Actor-Critic sees only the 64-dimensional output.

| Training setting | Representation Build | Task Performance |
| --- | --- | --- |
| Data | Random policy | Soft Actor-Critic's own experience after 1,000 random steps |
| Steps × seeds | 100,000 × 3 | 200,000 × 3 per condition |
| Optimizer | Adam, learning rate 1×10⁻⁴ | Adam, learning rate 3×10⁻⁴, γ = 0.99 |
| Batch | 256 (1.x); 32 (2.0); up to 16 clusters × (4 support + 4 query) (2.1–2.4) | 256 |
| Buffer | 70,000 frames | 200,000 transitions |
|  |  |  |

### 6.2 Evaluation criteria

Every metric is also computed for a **floor**: untrained encoders with the same architecture, averaged over 3 random initializations. A result counts only if it clears the floor in all three seeds.

**Health (main criteria for Representation Build).** Health is measured on 5,000 held-out random-policy steps, collected with a different seed from training.

1. **Held-out matching accuracy.** The training task is replayed with no gradient: up to 16 clusters with 4 support and 4 query frames each, and each query is assigned to its nearest prototype. The *fresh* version refits centroids on the held-out data, which tests the general skill. The *persisted* version reuses the centroids saved at the end of training, which tests whether the learned categories carry over to new transitions. A good result is well above the floor and close to training accuracy. A large gap below training accuracy means overfitting.
2. **Separation ratio.** This is the mean embedding distance between frames in different clusters divided by the mean distance within the same cluster, over up to 20,000 random pairs. Values above 1 mean the clusters are separated; the criterion is to beat the floor.
3. **Utili****z****ation and dead clusters.** Utilization is the normalized entropy of cluster or prototype usage, H / log K, which runs from 0 to 1. A dead cluster is one with no members. A healthy run has utilization near 1 and no dead clusters. Utilization near 0 with K−1 dead clusters means the representation has collapsed.
4. **State coherence.** This is the only health check against ground truth. For each state dimension, there is horizontal and vertical displacement, horizontal and vertical velocity, angle of rotation and angular velocity, it measures the share of variance explained by cluster membership. Overall coherence is the mean over the six dimensions.

A score of 0 means the clusters are unrelated to the physics, and 1 means cluster membership fixes the state. The per-dimension values show *which* quantities are captured.

The Yarats-style variants (1.x) use equivalent checks: *next-step agreement* (whether the online encoder plus predictor picks the prototype the target encoder gives the next frame), *augmentation invariance* (whether two random shifts of a frame are each other's nearest neighbor in a batch of 64) and mean cosine similarity between embeddings (≈ 1 signals collapse).

**Linear probe (supporting).** A logistic regression predicts the situation class, and a linear regression predicts each state dimension. Train and test sets are split by whole episodes. Scores are compared with the floor and with a linear probe on the true state. Probe accuracy separates encoders poorly because 79% of random-policy frames are "descent": always guessing the majority class already scores about 0.79. The per-dimension coefficient of determination is more informative.

**Task Performance.**

- **Mean return** over 100 deterministic episodes per trained agent.
- **Success rate:** the share of episodes with return ≥ 200.
- **Steps to solve:** the first step at which the 100-episode moving average reaches 200.
- **Learning curve:** the 100-episode moving average in 10,000-step buckets.

`trained` must beat `random_floor` for the representation to count as useful, and `true_state` is the reference ceiling. With 3 seeds, a difference between conditions counts only if their seed ranges do not overlap.

## 7. Results

By the health criteria, the Representation Build experiments produced encoders whose clusters generalize to held-out data and explain up to 60% of the variance in position and velocity. The Task Performance experiments showed that those encoders did not help Soft Actor-Critic land. Figures are 3-seed means ± standard deviation from the result files, unless a row says otherwise.

### 7.1 Representation Build experiments

**Purpose.** These experiments answer the first further question in the Introduction: *can a prototypical objective, trained without reward or labels, build a pixel representation that carries the lander's physical state?*

**What the experiment does.** A random policy plays Lunar Lander and fills a 70,000-frame buffer. The encoder (§6.1) is trained for 100,000 steps on one prototypical objective and never sees the reward or the situation labels. At step 100,000 it is evaluated with the health criteria in §6.2 on held-out frames, against untrained floors. The variants form a sequence, and each change responds to the previous result:

| Variant | Change | Reason |
| --- | --- | --- |
| 1.0 / 1.1 | Yarats-style Sinkhorn prototypes on raw frame stacks | Start from the published method \[2\] |
| 1.2 | Same objective on frame differences | Focus the encoder on motion, not the static scene |
| 2.0 | Instance-level prototypical loss | Replace the prototype bank with Snell-style episodes |
| 2.1 / 2.2 | Prototypical loss over k-means clusters of frame differences | Give episodes meaningful pseudo-classes (motion regimes) |
| 2.3 | 2.2 plus unit-length embeddings | Stop the loss being lowered just by inflating embedding size, a weakness found when reviewing 2.2 |
| 2.4 | Clusters of appearance (raw frames) instead of motion | Test whether motion is the right clustering signal |

**Results.** Table 2 gives the health metrics as 3-seed means.

**Table 2. Health of the Representation Build encoders at 100,000 steps****.**

| Variant | Held-out matching accuracy | Separation ratio (actual versus floor) | Utilisation / dead | State coherence |
| --- | --- | --- | --- | --- |
| 1.1 | Next-step agreement 0.487 (0.47–0.52) | — | 0.79 / 1–6 of 32 | Not measured |
| 1.2 | Seed 1: agreement 0.50. Seeds 0, 2: 0.00 | — | Seed 1: 0.83 / 2 of 32. Seeds 0, 2: 0.00 / 31 of 32 | Seed 1: 0.351. Seeds 0, 2: 0.00 (collapsed; cosine similarity 1.00) |
| 2.2 | Fresh 0.724 versus 0.182; persisted **0.876** versus 0.185 | **2.16** (1.84–2.35) versus 1.67 | 0.75 / **0 of 16** | **0.393** (0.346–0.454) |
| 2.3 | Persisted **0.974** (0.972–0.977) | **3.5–4.9** versus 1.7–2.0 | — / **0 of 16** | **0.424** (0.397–0.458) |
| 2.4 | Fresh 0.818 versus 0.905; persisted 0.813 versus 0.854 | 1.73 versus 1.69 | 0.95 / 0 of 16 | 0.177 (0.134–0.210) |

The 2.3 figures come from the variant's written summary, because its result files were removed. Its health evaluation differs from 2.2's, so floors are only roughly comparable.

- **Motion clustering (2.2, 2.3) passed every health criterion.** Persisted held-out matching was 0.876 (2.2) and 0.974 (2.3), 4–5× the floor. In 2.3 it nearly matched training accuracy (0.978–0.981), so the learned categories generalize to new data. Separation beat the floor in every seed, and no cluster died.
- **The motion clusters track the physics, but only translation.** In 2.2, cluster membership explains 60% of the variance in vertical position, 49% in vertical velocity, 47% in horizontal position and 44% in horizontal velocity. It explains only 25% in angle and 10% in angular velocity. The clustering feature takes the absolute value of the frame difference, which removes the direction of motion, and that is the likely reason rotation is lost.
- **2.2's health had weaker spots.** Fresh-centroid matching (0.724) was clearly below persisted matching (0.876), so the encoder learned its own 16 categories better than the general skill of grouping motion. Utilization was 0.71–0.79, so some clusters were used much more than others.
- **Appearance clustering (2.4) failed the health checks.** Held-out matching fell *below* the untrained floor (0.818 against 0.905) even though training accuracy was about 0.998, so the encoder overfitted the training buffer. Separation was at the floor, and coherence (0.177) was less than half of 2.2's.
- **Sinkhorn prototypes were fragile.** On frame differences (1.2) they collapsed in 2 of 3 seeds: 31 of 32 prototypes died, and every embedding pointed the same way. On raw frames (1.1) they stayed alive (utilisation 0.79), but next-step agreement was below 0.5. Two augmentations of the same frame landed on the same prototype only 43–47% of the time.

The linear probe is supporting evidence only. Its ordering broadly agrees with health: situation accuracy was 0.958 (1.1), 0.936 (2.2), 0.884 (2.3) and 0.792 (2.4, the majority-class rate). Where the two disagree, health is the more reliable guide. 2.3 scores below 2.2 on the probe but above it on every health metric, and the probe cannot flag overfitting or collapse the way held-out matching and utilization do.

*So, can a prototypical objective, trained **without reward or labels**, build a pixel representation that carries the lander's physical state?*

**A****ns****w****er****: partly.**

*Where it succeeded.* A prototypical loss over k-means motion clusters, trained with no reward or labels, built a representation that meets every health criterion:

- Its categories generalise to unseen transitions: persisted matching was 0.876 in 2.2 and 0.974 in 2.3, against a floor of about 0.19.
- They are well separated: 2.16 and 3.5–4.9, above the floor in every seed. All 16 clusters stay in use.
- Most importantly, they track real physics. Cluster membership explains 44–60% of the variance in horizontal and vertical position and velocity.

The success comes from the pseudo-labels. Frame differences group transitions by how the lander is moving, and moving differently usually means being in a different physical state. So learning to tell the clusters apart pushes the encoder to capture position and speed.

*Where it failed.* The representation does not carry the whole state:

- Clusters explain only 25% of angle variance and 10% of angular-velocity variance. The clustering feature takes the absolute value of the frame difference, so it cannot tell clockwise from anticlockwise rotation.
- Fresh-centroid matching (0.724) trails persisted matching (0.876). The encoder learned its own categories better than the general skill of grouping motion.
- The method depends on the choice of signal. Appearance clusters (2.4) overfitted and fell below the untrained floor on held-out matching. Sinkhorn prototypes on frame differences (1.2) collapsed in 2 of 3 seeds.

In short, a reward-free prototypical objective can build a physically meaningful representation, but only of the quantities its pseudo-labels can see.

### 7.2 Task Performance experiments

**Purpose.** These experiments answer the second further question: *does a pre-trained representation help a Soft Actor-Critic agent perform the task when it is used as a frozen input?*

**What the experiment does.** Each Representation Build checkpoint (step 100,000) is frozen inside an environment wrapper. On every step the wrapper renders the frame, takes the difference from the previous frame, passes it through the encoder and hands Soft Actor-Critic the 64-dimensional embedding as its observation. Soft Actor-Critic then trains for 200,000 steps. Three conditions are compared, each with the same seed set:

- **trained:** the pre-trained encoder.
- **random\_floor:** the same architecture, untrained.
- **true\_state:** Soft Actor-Critic on the raw 8-dimensional state, as the ceiling.

Variant 2.2 uses the Representation Build 2.2 encoders. 2.3 uses the 2.3 encoders, widens the actor and fixes torch seeding. 2.4 fine-tunes a pre-trained motion encoder through the critic loss instead of freezing it, and is still running.

**Results.**

**Table 3. Final performance over 100 deterministic evaluation episodes after 200,000 Soft Actor-Critic steps.**

| Sweep | Condition | Mean return (seed range) | Success rate |
| --- | --- | --- | --- |
| 2.2 (encoder from Representation Build 2.2) | trained | −626.1 (−836 to −226) | 0.00 |
|  | random\_floor | −179.3 (−275 to −132) | 0.00 |
|  | true\_state | **−31.1** (−53 to −6) | 0.01 |
| 2.3 (wider actor, fixed seeding) | trained | −431.9 (−558 to −346) | 0.00 |
|  | random\_floor | −185.6 (−294 to −132) | 0.00 |
|  | true\_state | **38.1** (−88 to 213) | 0.23 |

The pre-trained encoder was the worst condition in both sweeps. Its seed ranges do not overlap with `random_floor` in 2.3, though they overlap in 2.2. Its learning curves also fell over training. In 2.3 the 100-episode average went from about −260 to −290 in the first 10,000 steps to about −300 to −510 by 200,000. No condition reached a 100-episode average of 200. Only one `true_state` seed solved the task (2.3 seed 0: 213 ± 51, success rate 0.69).

**Why the results suffered.** The code and the health results point to four causes, in rough order of weight:

1. **Rotation is missing.** Health showed that the clusters explain only 25% of angle variance and 10% of angular-velocity variance, because the absolute-value clustering feature removes the direction of motion. Lunar Lander is lost by tipping over. A policy that cannot tell which way the lander is rotating cannot fire the correct side engine.
2. **Training data differs from use.** The encoder was fitted only on random-policy frames, which are mostly fast, tumbling descents. As Soft Actor-Critic improves, it visits hovering and slow, upright approaches that are rare in that data. That fits the learning curves getting worse as training went on.
3. **Compression removes detail that random features keep.** An untrained convnet is a random projection that keeps most of the input's information. Prototype training deliberately compresses frames into 16 motion categories and throws away within-cluster detail that the critic needs for fine value estimates. That is why `random_floor` can beat `trained` even though `trained` wins every health metric.
4. **The agent cannot adapt the encoder, and its budget is small.** The encoder is frozen, so critic gradients cannot recover missing features. Task Performance 2.4, now running, tests this by fine-tuning the encoder. And 200,000 steps is short for this task: `true_state` solved it in only 1 of 6 seeds. Two `random_floor` seeds in both sweeps score exactly −131.5 ± 44.0, which suggests they settled into the same degenerate deterministic behavior. That floor should be checked before it is relied on.

*S**o**, **d**oes such a pre-trained representation help a Soft Actor-Critic agent **perform the task** when it is used as a frozen input?*

**Answer****:**** ****No.** **not ****yet.**&#32;

As a frozen input, the pre-trained encoder made Soft Actor-Critic perform worse than an untrained encoder in both sweeps, and no trained-encoder episode reached a successful landing. The pre-training produced motion categories that are well organised and generalise to new data. But those categories leave out what the controller needs most: which way the lander is rotating, and the fine detail within each category. The representation is structured, but a policy cannot act on it well. Whether it helps once rotation is captured and the encoder is fine-tuned is still open, and Task Performance 2.4 is the first test of the second part.

### 7.3 What worked and what did not

Overall verdict: the prototypical objectives succeeded as representation learners for translation and speed, but failed as a route to better control.

**What worked**

- **Motion-based pseudo-classes produced healthy, generalising clusters.** In 2.2 and 2.3, persisted held-out matching was 0.876 and 0.974, 4–5× the floor. Separation beat the floor in every seed, and no cluster died.
- **The clusters track real physics.** 2.2's clusters explain 39% of state variance overall, and 44–60% for horizontal and vertical position and velocity. The linear probe agreed, with a coefficient of determination of 0.54–0.74 on those dimensions.
- **Yarats-style prototypes stayed alive on raw frames.** In 1.1, utilisation was 0.79 with only 1–6 of 32 prototypes unused.
- **The evaluation design caught false positives.** The floors showed that 2.4's clean t-distributed stochastic neighbour embedding islands and 0.998 training accuracy were not learning: held-out matching was below the untrained encoder's.
- **The Soft Actor-Critic set-up can solve the task.** One `true_state` seed reached 213 ± 51 with a 0.69 success rate, so the control failures on embeddings are not just a bug in the agent.

**What did not work**

- **The pre-trained encoder hurt control.** It was the worst condition in both sweeps: −626 and −432 mean return, against −179 and −186 for an untrained encoder. Its learning curves fell over training, and no episode was a success.
- **Rotation was not captured.** In 2.2, cluster membership explained only 25% of angle variance and 10% of angular-velocity variance, because taking the absolute value of the frame difference removes the direction of motion.
- **The general matching skill was weaker than the learned categories.** In 2.2, fresh-centroid matching was 0.724 against 0.876 with the saved centroids, and utilisation of 0.71–0.79 shows some clusters were used much more than others.
- **Some objectives collapsed or learned the wrong thing.** 1.2 (frame differences with Sinkhorn prototypes) collapsed in 2 of 3 seeds, with 31 of 32 prototypes dead. 2.4 (appearance clusters) overfitted: held-out matching fell below the untrained floor (0.818 against 0.905), and its clusters explained only 18% of state variance.
- **Baselines were weak.** Only 1 of 6 true-state Soft Actor-Critic seeds solved the task within 200,000 steps, which leaves little room to show a representation gain.

## 8. Limitations and Conclusion

### 8.1 Limitations

- Only 3 seeds per arm (1 in the pixel auxiliary-loss runs), and short budgets for pixel-based Soft Actor-Critic (100,000–500,000 steps) compared with published pixel-based results.
- Soft Actor-Critic hyperparameters were not tuned for this environment. Even true-state Soft Actor-Critic solved the task in only 1 of 6 seeds within 200,000 steps.
- The situation labels are hand-set rules, and the "ceiling" probe on the true state is linear, so it is not a true upper bound (Representation Build 2.2 scores 1.53 on the normalised scale).
- Evaluation code differs between variants. Some evaluations are unseeded (fixed in 1.1 and flagged again in the 2.2 code review), and in 2.2 the diff embedding is labelled with the pre-transition situation. Numbers from different variants are only roughly comparable.

### 8.2 Conclusion

Prototypical objectives in this project learned the structure they were asked to learn, but that structure did not reliably make a reinforcement learning agent more robust or more capable on Lunar Lander. As a supervised auxiliary loss, the prototype head's effect on noise-robustness is inside seed variance, and it lowered return. As a reward-free pretext task, clustering frame differences gave the healthiest and most state-informative encoder in the project (0.88 held-out matching, 39% of state variance explained). Freezing that encoder under Soft Actor-Critic still did worse than a random encoder. The main lesson is that cluster health and probe scores are weak stand-ins for usefulness in control, and the downstream test is needed. But they were useful for diagnosis. They separated real learning from artefacts (2.4's islands), caught collapse early (1.2), and showed exactly which state was missing (rotation), which explains the control failure and points to the fix.

### 8.2 Future Work

The next steps below are ordered by expected value per unit of compute:

1. **Fine-tune instead of freezing.** Complete Task Performance 2.4, which initialises Soft Actor-Critic's encoder from the pre-trained weights and lets critic gradients through, as in the downstream phase of Yarats et al. \[2\]. Compare it with training from scratch at equal steps.
2. **Fix the rotation gap.** Keep the sign of the frame difference when building clustering features instead of taking its absolute value, as proposed in the 2.2 code review. Then check whether angle and angular-velocity coherence improve, and whether tilt control improves.
3. **Add a standard baseline.** Image augmentation on the pixel-based Soft Actor-Critic agent, as in Data-regularised Q-learning or Contrastive Unsupervised Representations for Reinforcement Learning, would show whether the pixel pipeline can learn at all at these budgets before prototype variants are judged against it.
4. **Consolidate evaluation**: merge the scripts into one seeded, shared module so that numbers are comparable across variants.

## References

1. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems*. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
2. Yarats, D., Fergus, R., Lazaric, A., & Pinto, L. (2021). Reinforcement learning with prototypical representations. *International Conference on Machine Learning*. [arXiv:2102.11271](https://arxiv.org/abs/2102.11271)
3. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *International Conference on Machine Learning*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
4. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., & Joulin, A. (2020). Unsupervised learning of visual features by contrasting cluster assignments. *Advances in Neural Information Processing Systems*. [arXiv:2006.09882](https://arxiv.org/abs/2006.09882)
5. Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in Neural Information Processing Systems*. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
6. Ciriello, D. (orobix). Prototypical Networks for Few-shot Learning in PyTorch. [GitHub](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)
7. Yarats, D. Official implementation of Reinforcement Learning with Prototypical Representations. [GitHub](https://github.com/denisyarats/proto)