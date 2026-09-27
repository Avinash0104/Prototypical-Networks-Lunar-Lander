import ctypes
import os
import subprocess
import sys
import time
import traceback

from experiment.checkpoint import find_latest_checkpoint

SEEDS = (0, 1, 2)
NUM_STEPS = 100_000
LOGSTEP = 5_000
RESULT_DIR = "results"
LOG_PATH = "training_log.txt"
RETRY_DELAY = 10
MAX_RETRY_DELAY = 300


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging():
    log = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)


def keep_awake():
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: stop Windows idle-sleeping while the runner lives
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def latest_step(seed):
    _, step = find_latest_checkpoint(f"{RESULT_DIR}/prototype_seed{seed}")
    return step


def is_seed_done(seed):
    return latest_step(seed) >= NUM_STEPS


def worker(seed):
    from experiment.training import train
    train(num_steps=NUM_STEPS, seed=seed, device="cuda", result=RESULT_DIR, resume=True, logstep=LOGSTEP)


def main():
    keep_awake()
    delay = RETRY_DELAY
    while True:
        pending = [seed for seed in SEEDS if not is_seed_done(seed)]
        if not pending:
            break
        # one attempt per pending seed per pass, so a seed that keeps failing can't starve the others
        for seed in pending:
            try:
                before = latest_step(seed)
                print(f"=== seed {seed}: starting worker (checkpoint step {before}) ===", flush=True)
                # each attempt is its own process, so a CUDA fault, OOM kill or segfault
                # can't leave the runner itself in a broken state
                code = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker", str(seed)]).returncode
                if is_seed_done(seed):
                    print(f"=== seed {seed} complete ===", flush=True)
                    delay = RETRY_DELAY
                    continue
                progressed = latest_step(seed) > before
                delay = RETRY_DELAY if progressed else min(delay * 2, MAX_RETRY_DELAY)
                print(f"[seed {seed}] worker exited with code {code} at step {latest_step(seed)}, retrying in {delay}s", flush=True)
            except Exception:
                traceback.print_exc()
                delay = min(delay * 2, MAX_RETRY_DELAY)
                print(f"[seed {seed}] runner error, retrying in {delay}s", flush=True)
            time.sleep(delay)
    print("=== all seeds complete ===", flush=True)


if __name__ == "__main__":
    setup_logging()
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
