# Prototype Arm: Task-Performance Issues

Findings from the 500k-step, 3-seed run, evaluated via `evaluation/returns.py` and `evaluation/probe.py` at `checkpoint_step=500_000`.

## Results

**Task performance** (30-episode deterministic rollouts, averaged over 3 seeds):

| Metric | Value |
|---|---|
| train_mean | -166.2 |
| shift_mean | -56.4 |

Both the worst raw task performance and the largest train/shift gap of any arm evaluated in this study.

**Representation robustness** (linear-probe accuracy drop under noise-distribution shift — lower is better):

| Metric | Value |
|---|---|
| probe_train_acc | 0.960 |
| probe_shift_acc | 0.664 |
| degradation | 0.296 |

Robustness is strong — the best of any arm evaluated. The task-performance cost below is the trade-off for that.

## Issues

**1. Unbounded embedding scale lets the loss cheat.**
No normalization or temperature bounds the squared Euclidean distances used in the loss. Since each prototype is just the mean of `support_z` for its class, scaling *all* embeddings by a constant `c` scales every pairwise squared distance by `c²` — sharpening the softmax and lowering cross-entropy loss without any real improvement in class separability. Gradient descent has a standing incentive to inflate output scale rather than learn better structure. That scale pressure propagates back through the shared `Encoder`, pushing `z` — which the critic also depends on for Q-value regression — into a regime not suited for value estimation.
- Diagnostic: log `support_z.norm(dim=-1).mean()` over training; check for unbounded growth.
- Mitigation: L2 penalty on embedding norm, or a learned scale/temperature parameter.

**2. Two extra forward passes on a separately-sampled batch feed gradient into the shared encoder every step.**
Every training step, `support_obs` and `query_obs` (336 examples total, class-stratified) pass through `encoder` + `projection_head`, and the resulting loss backpropagates into the same encoder the critic uses. This class-balanced sample has a very different composition from the critic's naturally imbalanced IID replay batch (mostly "descent"), so the encoder is pulled by two differently-shaped data distributions every step.

**3. Rare-class duplication adds noisy gradient signal.**
`sample_episode` uses `replace=True` when a class has fewer than `n_support + n_query` (84) examples in the buffer. Early in training — and for "contact" especially, since landings are rare before the policy improves — the same handful of transitions get drawn repeatedly as both support and query. A prototype built from duplicated points, scored against overlapping query points, is a noisy, low-information gradient signal.

**4. No gradient-magnitude balancing beyond the scalar `lambda_t`.**
`total_loss = critic_loss + lambda_t * aux_loss` is a simple weighted sum through one shared optimizer. `lambda_t` caps at 1.0, but nothing equalizes *gradient magnitude*. If the unbounded distance loss (issue 1) produces larger gradients w.r.t. encoder parameters than the bounded MSE critic loss, it can dominate the actual parameter update even at nominal parity.

## Suggested next steps

1. Add embedding-norm logging to confirm/refute issue 1 (cheapest, highest-value check).
2. If confirmed, try an L2 penalty on `support_z`/`query_z` norm, or a learned scale parameter, and rerun.
3. Taper or cap `lambda_t` below 1.0 rather than letting it reach full parity with the critic loss.
4. Guard `sample_episode` against support/query overlap even under `replace=True`.
