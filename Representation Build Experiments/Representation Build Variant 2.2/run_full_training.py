import time
import traceback

from experiment.training import train
from experiment.checkpoint import find_latest_checkpoint

SEEDS = (0, 1, 2)
NUM_STEPS = 100_000
RESULT_DIR = "results"


def is_seed_done(seed):
    run_dir = f"{RESULT_DIR}/prototype_seed{seed}"
    _, step = find_latest_checkpoint(run_dir)
    return step >= NUM_STEPS


def run_seed(seed):
    attempt = 0
    while not is_seed_done(seed):
        attempt += 1
        try:
            print(f"[seed {seed}] attempt {attempt}", flush=True)
            train(num_steps=NUM_STEPS, seed=seed, device="cuda", result=RESULT_DIR, resume=True)
        except Exception:
            print(f"[seed {seed}] crashed on attempt {attempt}, resuming in 10s", flush=True)
            traceback.print_exc()
            time.sleep(10)


def main():
    for seed in SEEDS:
        if is_seed_done(seed):
            print(f"[seed {seed}] already complete, skipping", flush=True)
            continue
        print(f"=== starting seed {seed} ===", flush=True)
        run_seed(seed)
        print(f"=== seed {seed} complete ===", flush=True)
    print("=== all seeds complete ===", flush=True)


if __name__ == "__main__":
    main()
