#!/bin/bash
# Keeps relaunching `python -m evals.runner` no matter how or why it dies --
# crashes, exceptions, OOM kills, or anything else that ends the process --
# until the full training+eval sweep is verifiably complete. Every individual
# run is already resumable from its own checkpoint (experiment/training.py),
# so a relaunch picks back up within one checkpoint interval of wherever it
# was killed, never from scratch.
set -uo pipefail

is_done() {
    python -c "
import json, os, sys
path = os.path.join('results', 'eval_results.json')
if not os.path.exists(path):
    sys.exit(1)
try:
    with open(path) as f:
        data = json.load(f)
    rows = data.get('rows', [])
    combos = {(r['arm'], r['seed']) for r in rows}
    expected = {(arm, seed) for arm in ('trained', 'random_floor', 'true_state') for seed in (0, 1, 2)}
    sys.exit(0 if expected.issubset(combos) else 1)
except Exception:
    sys.exit(1)
"
}

attempt=0
while true; do
    if is_done; then
        echo "[supervisor] full sweep confirmed complete (results/eval_results.json present and valid) -- stopping."
        break
    fi
    attempt=$((attempt + 1))
    echo "[supervisor] launching evals.runner -- attempt $attempt -- $(date)"
    python -m evals.runner
    code=$?
    echo "[supervisor] evals.runner exited with code $code at $(date) -- will check completion and relaunch if needed"
    sleep 5
done
