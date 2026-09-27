# Representation Build Variant 2.2 — Issues & Suggested Fixes

Findings from a code review of this experiment (encoder, prototypical loss, clustering,
buffer, training loop, evaluation scripts) plus the already-completed 3-seed,
100k-step run in `results/`. None of these are crashes or bugs that break the code —
they're risks or real limitations worth knowing about.

## Issues

**1. Nothing stops the embeddings from just "getting bigger" to cheat the loss.**
The loss works by measuring distance between points in embedding space. If the model
just scales all its outputs up (bigger numbers, not better structure), the distances
get bigger too, which can fool the loss into thinking it's doing better than it really
is. Nothing in `models/prototype.py` caps this. The training numbers from this run look
healthy (accuracy leveled off around 85-93%, didn't shoot up to 100%), so it probably
didn't happen here — but it's not guarded against, so it could happen on a different
run/seed.

**2. A small chance of duplicate samples sneaking into training.**
Each training batch first picks ~1,000 candidate frames at random, allowing the same
frame to be picked more than once by chance (`ReplayBuffer._random_valid_indices`
samples with replacement). Statistically, out of 1,000 random picks from a big pool,
you'd expect a handful of accidental repeats. It's a minor, spread-out effect — not
concentrated on any one class the way the old `replace=True` rare-class issue in
variant 2.1 was — so low impact, but worth knowing it's there.

**3. One evaluation number is misleading if read at face value.**
In `results/eval_probe.txt`, the model's representation scores *higher* (0.94) than the
"ceiling" (0.86) it's supposedly being measured against. That sounds impossible, but
it's not a bug — the "ceiling" is just a simple straight-line (linear) classifier on the
raw physics numbers, and the real classification boundary (`compute_situation`) isn't a
straight line. The model's learned representation is better at reshaping the data than
that simple linear ceiling is at reading it. Not an error, just a confusingly-named
number — don't read `normalized_score > 1` as broken.

**4. The representation doesn't seem to capture rotation speed at all.**
Across all 3 seeds and both evaluation scripts (`evaluation/probe.py`'s R²,
`evaluation/health.py`'s explained variance), the ability to recover the lander's
angular velocity (how fast it's spinning) from the learned representation is near zero
— sometimes worse than a random, untrained model. Position and linear velocity
(x, y, vx, vy) are captured well; spin rate isn't. This is a real, consistent gap in
what the representation learned, not a code error.

**5. `probe.py` and `visual.py` label each motion with the situation before it
happened, not after.** Both scripts compute a diff spanning from frame `t` to frame
`t+1`, but tag it with the situation label from frame `t` (before the motion), not
frame `t+1` (what the motion actually produced). Concretely, the exact frame-diff that
captures a touchdown gets labeled `critical` — whatever the situation was one step
earlier — never `contact`. This compounds with the already-rare `contact` class: it's
not just short on examples, the few it does have are the least informative ones
(already-landed, motionless frames), while the actually distinctive touchdown-motion
examples get siphoned into `critical` instead.

**6. Re-running `probe.py` gives different numbers every time, for a reason that's
easy to miss.** `collect_embeddings` seeds `env.reset(seed=seed)` but never seeds
`env.action_space.sample()`, so every run walks a different random trajectory. This
isn't cosmetic: the `angle`/`ang_vel` R² values in `eval_probe.txt` swung meaningfully
across separate runs during this session's testing (e.g. `angle` R² moved from -0.01 to
+0.43 to +0.27 across three otherwise-identical invocations), and even the specific
probe-vs-linear-baseline numbers cited in Issue 3 will drift on the next run. The same
gap exists in `health.py`'s held-out buffer collection.

## Suggested fixes

### Issue 1: unbounded embedding scale

**Fix: L2-normalize the embeddings before computing distances.** If every embedding is
forced onto the unit sphere (length exactly 1), scaling has no effect anymore —
multiplying by a constant `c` doesn't change direction, and only direction is left. That
removes the shortcut entirely, for free, with no new hyperparameter.

In `models/prototype.py`:

```python
import torch
import torch.nn.functional as F

def prototypical_loss(support_z, support_y, query_z, query_y, num_classes):
    support_z = F.normalize(support_z, dim=-1)
    query_z = F.normalize(query_z, dim=-1)

    prototypes = torch.stack([support_z[support_y == c].mean(dim=0) for c in range(num_classes)])
    prototypes = F.normalize(prototypes, dim=-1)  # the mean of unit vectors drifts off the sphere -- renormalize it

    dists = torch.cdist(query_z, prototypes, p=2).pow(2)
    log_p = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p, query_y)

    with torch.no_grad():
        accuracy = (log_p.argmax(dim=1) == query_y).float().mean()

    return loss, accuracy
```

Caveat: fixing embeddings to a unit sphere also fixes how "sharp" the softmax can ever
be. If accuracy gets *worse* after this change, add one learnable, clamped temperature
so the model can still control sharpness, just within bounds instead of unbounded:

```python
# in Agent.__init__: self.log_temp = torch.nn.Parameter(torch.zeros(1, device=device))
# add self.log_temp to proto_optimizer's params

temp = log_temp.exp().clamp(max=100)  # bounded -- can't run away like raw norm could
dists = torch.cdist(query_z, prototypes, p=2).pow(2)
log_p = F.log_softmax(-dists * temp, dim=1)
```

### Issue 2: duplicate candidates from sampling with replacement

**Fix: draw without replacement from the actual set of valid indices**, instead of
drawing randomly with replacement and retrying collisions. Compute validity over the
whole buffer once, then sample from that set directly:

In `experiment/buffer.py`:

```python
def _random_valid_indices(self, n):
    latest = (self.ptr - 1) % self.capacity
    all_indices = np.arange(self.size)
    invalid = (all_indices == latest) | (self.steps_since_reset[(all_indices + 1) % self.capacity] == 0)
    valid_indices = all_indices[~invalid]
    return np.random.choice(valid_indices, size=n, replace=False)
```

This guarantees uniqueness rather than just making collisions statistically rare, and is
simpler than the current masked-retry-loop version (no `while mask.any()` needed).
Trade-off: it does `O(capacity)` work per call (recomputes the mask over the whole
buffer) instead of `O(n)`, but at numpy speed that's tens of microseconds -- across a
full ~99,000-call training run that's a few seconds of added overhead total, negligible
next to rendering/encoder compute.

### Issue 3: "ceiling" isn't a real upper bound

**Fix: rename it, don't recompute it.** `situation_label` is
`compute_situation(true_state)` -- a heuristic bucket the project itself defines, not an
independent ground truth. That makes "the true achievable ceiling" a degenerate question
(a formula can always recover itself exactly, which is tautological, not informative).
The actually useful thing the current number gives you is a same-model-class comparison
-- "how does a linear probe do on full state vs. on the learned embedding" -- which just
isn't a ceiling. Keep the computation exactly as-is, fix only the naming/framing.

In `evaluation/probe.py`:

```python
def baselines(num_steps=5000, seed=2000, num_floor_draws=3):
    floor_accs = []
    floor_r2s = []
    linear_state_probe_acc = None
    for _ in range(num_floor_draws):
        z, states, y, episode_ids = collect_embeddings(random_agent(), num_steps=num_steps, seed=seed)
        floor_acc, _ = fit_probe(z, y, episode_ids)
        floor_accs.append(floor_acc)
        floor_r2s.append(fit_regression_probe(z, states, episode_ids))
        if linear_state_probe_acc is None:
            linear_state_probe_acc, _ = fit_probe(states, y, episode_ids)

    avg_floor_r2 = {dim: float(np.mean([r[dim] for r in floor_r2s])) for dim in STATE_DIM_NAMES}
    return float(np.mean(floor_accs)), linear_state_probe_acc, avg_floor_r2
```

And update `main()`'s print statements to drop the "floor=0/ceiling=1" framing that
implies bounded semantics:

```python
print(f"{'floor (random encoder)':24s} acc={floor_acc:.3f}")
print(f"{'linear probe on true state':24s} acc={linear_state_probe_acc:.3f}")
...
print(f"Score relative to linear full-state probe (avg over {len(seeds)} seeds): {norm_avg:.3f} +/- {norm_std:.3f}")
print("  (can exceed 1.0 -- situation_label's boundary is nonlinear, so a linear full-state")
print("   probe isn't a strict upper bound, just a same-model-class reference point)")
```

If a genuine bounded ceiling is ever wanted, the honest way to get one is fitting a
*more expressive* classifier (a shallow decision tree or small MLP) on `true_state` --
something that can actually represent `compute_situation`'s threshold logic empirically,
rather than either a linear probe (too weak to reach it) or the literal formula
(tautological). Not necessary unless that specific property is wanted.

### Issue 4: angular velocity isn't captured

**Recommendation: leave this as a documented limitation for now, not something to fix
immediately.** Fixing it means changing what `diff_feature()` produces, which changes
the pseudo-labels the encoder is trained against -- that requires a full retrain (all 3
seeds, 100k steps each) just to find out whether it worked. The current checkpoints are
already in active use as the input to a separate downstream task-performance experiment
(Task Performance Variant 2.2); retraining now would orphan that in-progress work. The
better trigger for revisiting this is empirical, not preemptive: if that task-performance
comparison shows the `trained` condition specifically struggling with tilt/rotation
control relative to the `true_state` ceiling, that's the signal this gap actually matters
enough to pay for a retrain. If overall task performance is reasonable despite the gap,
it isn't worth the cost.

**If/when it is worth fixing**, the original root-cause diagnosis holds, but the fix
ordering below needs correcting. `diff_feature` takes `.abs()` of the pixel difference
*before* pooling -- that erases the sign of each pixel's change (which side got
brighter vs. darker) before any pooling happens at all. A rotation (an asymmetric,
signed pattern -- one side brightens, the other darkens) and a pure translation (a
same-magnitude "shifted blob" in abs-diff) can look similar once sign is gone,
*regardless of resolution*. That changes the recommended order:

- **(c) drop `.abs()` before pooling is the fix that actually addresses the root
  cause** -- try this first, not last. It's still a small change (`diff_feature` in
  `experiment/clustering.py`), and k-means works fine on signed real-valued features.
- **(b) increasing `pool_size` to 16 is a reasonable complement to (c), not a
  substitute for it.** Tried alone (as originally suggested here), it gives k-means a
  finer-grained picture of *how much* changed where, but the data feeding it has
  already had *which direction* erased -- it's unlikely to recover what (c) actually
  restores.
- **(a) widening the frame gap between diffed frames** remains the most invasive option
  (requires the buffer to store a short frame history instead of just consecutive
  frames) and should stay last -- only worth trying if (c)+(b) together still aren't
  enough.

One more thing worth checking if this is revisited: `ReplayBuffer.sample_by_cluster`'s
eligibility filter (`len(members) >= num_support + num_query`) could work against a
rotation-focused fix even after it lands -- if a "rotation" cluster is a small minority
under a random-action rollout, it may often fail this threshold and get excluded from
training batches entirely. Worth watching `dead_clusters`/`utilization` in `health.py`'s
output after the fix, and increasing `candidate_pool` if a rotation cluster turns out
sparse.

### Issue 5: diffs labeled with the wrong end of the transition

**Fix: label with the post-transition state/label, not the pre-transition one.** This
also simplifies both functions -- no need to track `prev_label`/`prev_state` separately
anymore, since the current `info` right after `env.step()` already describes what the
diff just produced.

In `evaluation/probe.py`:

```python
def collect_embeddings(agent, num_steps=5000, seed=2000):
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    prev_frame, info = env.reset(seed=seed)

    zs = []
    states = []
    labels = []
    episode_ids = []
    episode_id = 0

    for _ in range(num_steps):
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)

        obs_t = torch.from_numpy(prev_frame).float().unsqueeze(0)
        next_obs_t = torch.from_numpy(frame).float().unsqueeze(0)
        with torch.no_grad():
            z = agent.encoder(frame_diff(obs_t, next_obs_t))

        zs.append(z.squeeze(0).numpy())
        states.append(info["true_state"])       # post-transition -- describes what this diff produced
        labels.append(info["situation_label"])  # post-transition
        episode_ids.append(episode_id)

        prev_frame = frame
        if terminated or truncated:
            prev_frame, info = env.reset()
            episode_id += 1

    return np.array(zs), np.array(states), np.array(labels), np.array(episode_ids)
```

In `evaluation/visual.py`, the identical pattern:

```python
def collect_embeddings_labels_clusters(agent, centroids=None, num_steps=3000, seed=3000, pool_size=8):
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    prev_frame, info = env.reset(seed=seed)

    zs, labels, clusters = [], [], []
    for _ in range(num_steps):
        action = env.action_space.sample()
        frame, reward, terminated, truncated, info = env.step(action)

        obs_t = torch.from_numpy(prev_frame).float().unsqueeze(0)
        next_obs_t = torch.from_numpy(frame).float().unsqueeze(0)
        with torch.no_grad():
            z = agent.encoder(frame_diff(obs_t, next_obs_t))
        zs.append(z.squeeze(0).numpy())
        labels.append(info["situation_label"])  # post-transition

        if centroids is not None:
            feat = diff_feature(obs_t, next_obs_t, pool_size=pool_size).numpy()
            clusters.append(int(assign_clusters(feat, centroids)[0]))

        prev_frame = frame
        if terminated or truncated:
            prev_frame, info = env.reset()

    zs = np.array(zs)
    labels = np.array(labels)
    clusters = np.array(clusters) if centroids is not None else None
    return zs, labels, clusters
```

### Issue 6: `probe.py`'s rollout isn't reproducible

**Fix: seed the action space, not just the environment reset.** One line, in every
function that calls `env.action_space.sample()` for evaluation purposes:

```python
env = Environment(seed=seed)
env.action_space.seed(seed)
```

This applies to `evaluation/probe.py::collect_embeddings`,
`evaluation/visual.py::collect_embeddings_labels_clusters` (both already shown fixed
above, as part of the Issue 5 change), and `evaluation/health.py::_collect_held_out_buffer`:

```python
def _collect_held_out_buffer(num_steps, seed):
    env = Environment(seed=seed)
    env.action_space.seed(seed)
    ...
```

Deliberately **not** applying this to `experiment/training.py::train()` -- training
already resumes from arbitrary checkpoints with a fresh numpy RNG state each time, so
exact action-sequence determinism isn't a real goal there and wouldn't hold across a
resume boundary anyway. This fix is specifically for the evaluation scripts, whose
whole purpose is giving a stable, comparable read of representation quality across
repeated invocations.
