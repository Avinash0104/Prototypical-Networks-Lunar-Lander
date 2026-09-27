# ROLE

You are guiding me through building this project, phase by phase, in chat. At each step: explain what we're building and why, then give me the code for that step. I will be the one entering/running the code myself — don't assume you're executing anything. Go one phase at a time, wait for me to confirm a milestone is met (or to report a problem) before moving to the next phase. If something breaks, help me debug rather than jumping ahead. Keep track of which phase we're on.

Start by confirming I have Gymnasium, PyTorch, and Stable-Baselines3 (or ask what RL library I want to use — see Phase 1) installed, then begin at Phase 1.

---

# THE PROJECT

**Question:** does adding a supervised prototype loss to an RL agent's encoder make its state representations more robust to irrelevant/noisy observation dimensions, compared to a plain agent and an agent with an ordinary auxiliary classifier head?

**No domain randomisation, no vision.** This is a small, fast, fully vector-based experiment using Gymnasium, meant to isolate the core mechanism cheaply before (or instead of) doing it in a harder visual setting.

---

# TECHNICAL SPEC

## Environment

Base env: `LunarLanderContinuous-v3` (Gymnasium). True state is 8-dim (position, velocity, angle, angular velocity, leg contacts).

**Distractor wrapper.** Build a `gymnasium.Wrapper` that appends **32 noise dimensions** to the true 8-dim state, giving a 40-dim observation. Noise must be statistically independent of the true state and of the situation label (see below) — this independence is what makes it a valid distractor rather than a leaky feature. Implement it as a slow random walk per dimension (small independent step each `step()`, resampled fresh at `reset()`), since this is the closest analogue to slowly-drifting visual conditions. Keep a config flag to switch to pure i.i.d. Gaussian noise per step as an alternative/ablation.

**Situation labels.** Computed by the wrapper from the *true* state only (never from the distractor block), stored alongside each transition:
- 0 — High descent: altitude above ~0.6
- 1 — Approach: mid altitude, moderate velocity
- 2 — Critical: low altitude with high velocity or large tilt
- 3 — Contact: either leg touching

Pick concrete thresholds from the true state's known ranges and hardcode them; exact cutoffs aren't critical, consistency across all three arms is.

## Algorithm

**SAC**, off-policy, continuous actions (matches `LunarLanderContinuous-v3` and matches the algorithm family used in the vision-based literature this idea comes from). Use Stable-Baselines3's SAC as the base, with a custom features-extractor/encoder so the auxiliary loss can be attached — or a from-scratch minimal SAC if I say I'd rather implement it manually. Ask me which before Phase 2.

## Architecture

1. **Encoder:** MLP, 40 → 128 → 64 (z). Shared between the SAC actor/critic and the auxiliary module.
2. **SAC actor + critics** on top of z. Standard SAC continuous-action heads.
3. **Projection head** (prototype arm only): linear/2-layer MLP, z (64) → u (32), L2-normalised.
4. **Prototype bank** (prototype arm only): K = 4 learnable unit-norm vectors in R^32, one per situation, re-normalised to unit norm after every optimizer step. Assignment: softmax over (u · c_k) / τ, τ ≈ 0.1. Loss: `L_proto = cross_entropy(p, situation_label)`.
5. **Classifier head** (control arm only): plain 2-layer MLP, z (64) → 4 logits, cross-entropy on the same labels. No prototypes, no projection head, no unit-norm geometry.

## Objective

`L_total = L_SAC + λ(t) · L_aux`. λ = 0 for the first ~10k steps (warmup), then ramp linearly to 0.1 over the next ~10k steps, then hold. L_aux is L_proto, L_classifier, or absent (baseline), selected by config/flag. The auxiliary loss is computed on the same replay-buffer minibatches SAC already samples — the buffer must store `situation_label` alongside each transition.

## The three arms (this is the experiment)

1. **Baseline** — SAC only, no auxiliary loss.
2. **Prototype** — SAC + prototype loss (the method).
3. **Classifier control** — SAC + plain classifier head, same labels, same λ. This isolates whether prototype *geometry* matters or whether any auxiliary supervision would do.

Run each arm with 3–5 seeds, ~300k–500k environment steps.

## Evaluation

1. **Learning curves.** Episode return vs. steps, under the training distractor condition, for all three arms, all seeds.
2. **Distribution-shift test.** After training, evaluate all frozen agents (no more learning) under a harder distractor condition — e.g., doubled noise-walk step size, or i.i.d. noise instead of the walk. Compare return degradation across arms.
3. **Linear probe.**
   - Roll out each frozen trained agent for some episodes under the *training* distractor condition; collect (z, situation_label) pairs.
   - Do the same under the *shifted* distractor condition.
   - Fit a linear classifier (logistic regression) on the training-condition (z, label) pairs only — no gradients into the encoder.
   - Evaluate that same fitted probe's accuracy on the shifted-condition embeddings.
   - Compare probe accuracy (training-condition vs shifted-condition) across the three arms. A smaller drop = more invariant representation.
4. **Embedding visualisation.** UMAP or t-SNE of z from rollouts, coloured by situation label, one plot per arm. Expect tighter/more separated clusters for the prototype arm if the effect is real.

## Codebase layout

```
project/
├── env/
│   └── distractor_wrapper.py     # appends noise, computes situation_label
├── models/
│   ├── encoder.py                 # 40 -> 128 -> 64
│   ├── prototype.py               # projection head + bank + loss (prototype arm)
│   └── classifier_head.py         # control arm
├── training/
│   ├── buffer.py                   # if not using SB3's built-in buffer, must store situation_label
│   └── train.py                    # wires L_total, the lambda schedule, arm selection
├── evaluation/
│   ├── evaluate.py                 # learning curves, shifted-condition return test
│   ├── probe.py                    # linear probe fit + evaluate
│   └── visualise.py                # UMAP/t-SNE plots
├── configs/
│   ├── baseline.yaml
│   ├── prototype.yaml
│   └── classifier.yaml
└── results/
```

---

# BUILD PHASES — GUIDE ME THROUGH THESE IN ORDER

**Phase 1 — Environment.** Confirm library choices (SAC via Stable-Baselines3 vs. manual implementation). Build `distractor_wrapper.py`: noise dimensions, situation label computation, config flag for walk-vs-iid noise. Milestone: wrapped env resets/steps correctly, observation shape is (40,), and printing labels across a random rollout shows sensible transitions (0→1→2→3 as the lander descends).

**Phase 2 — Baseline agent.** Set up SAC with the custom encoder (40 → 128 → 64) but no auxiliary loss. Train on the wrapped env. Milestone: episode return improves over training steps; log this to confirm the pipeline works end to end before adding any complexity.

**Phase 3 — Method + control.** Build `prototype.py` and `classifier_head.py`. Wire `L_total` with the lambda warmup/ramp schedule into the SAC update. Add config-based arm selection. Milestone: on a short run, prototype loss decreases and prototype assignments roughly track the true situation labels; classifier loss decreases similarly.

**Phase 4 — Full runs.** Train all three arms × 3–5 seeds to ~300–500k steps. Milestone: learning curves collected for all arms/seeds, saved checkpoints for every run.

**Phase 5 — Shift test + probe.** Build the harder distractor condition, evaluate all frozen agents on it, and implement the linear probe (fit on training-condition embeddings, test on shifted-condition embeddings). Milestone: a table of return-degradation and probe-accuracy-degradation per arm.

**Phase 6 — Visualisation + writeup.** Generate UMAP/t-SNE plots per arm, colored by situation label. Assemble the final comparison: learning curves, shift-test table, probe-accuracy table, embedding plots.

---

# HOW TO GUIDE ME

- One phase at a time — explain the concept for that phase briefly, then give me runnable code.
- Tell me exactly what output or number proves the milestone is met before we move on.
- When I report results or paste errors, diagnose using the spec above (check shapes, check the label computation, check gradients aren't leaking into a frozen encoder during the probe step, check the lambda schedule).
- Point out explicitly if anything I built violates the "same everything except one loss" comparison between arms — that's the whole experiment.