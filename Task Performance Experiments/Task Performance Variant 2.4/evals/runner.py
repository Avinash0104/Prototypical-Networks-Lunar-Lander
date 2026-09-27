import time
import traceback

from experiment.training import train
from experiment.checkpoint import find_latest_checkpoint
from evals.evaluate import main as evaluate_main

CONDITIONS = ("trained", "random_floor", "true_state")
SEEDS = (0, 1, 2)
NUM_STEPS = 200_000
RESULT_DIR = "results"


def is_run_done(arm, seed):
    run_dir = f"{RESULT_DIR}/{arm}_seed{seed}"
    _, step = find_latest_checkpoint(run_dir)
    return step >= NUM_STEPS


def run_condition_seed(arm, seed, device="cpu"):
    attempt = 0
    while not is_run_done(arm, seed):
        attempt += 1
        try:
            print(f"[{arm} seed{seed}] attempt {attempt}", flush=True)
            train(num_steps=NUM_STEPS, seed=seed, arm=arm, result=RESULT_DIR, resume=True, device=device)
        except Exception:
            print(f"[{arm} seed{seed}] crashed on attempt {attempt}, resuming in 10s", flush=True)
            traceback.print_exc()
            time.sleep(10)


def main(device="cpu"):
    for arm in CONDITIONS:
        for seed in SEEDS:
            if is_run_done(arm, seed):
                print(f"[{arm} seed{seed}] already complete, skipping", flush=True)
                continue
            print(f"=== starting {arm} seed{seed} ===", flush=True)
            run_condition_seed(arm, seed, device=device)
            print(f"=== {arm} seed{seed} complete ===", flush=True)

    print("=== all training runs complete -- evaluating ===", flush=True)
    evaluate_main(conditions=CONDITIONS, seeds=SEEDS, result=RESULT_DIR)
    print("=== all done ===", flush=True)


if __name__ == "__main__":
    main(device="cuda")