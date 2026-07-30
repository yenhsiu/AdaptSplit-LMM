"""
Quick, intuitive check of whether the MME-derived routing threshold
(base_lora <= N=234 -> original) roughly holds on VQA-v2 too.

Method (deliberately simple, per discussion -- not the full McNemar's-test
apparatus used for the original MME calibration):
  - Pure N effect only: --no-quant --n-tokens N --merge --force-model X
  - 6 fixed N points (50, 150, 250, 350, 450, 550) spanning the range
  - All 3 checkpoints (base_lora, plus_lora, original) -- re-testing
    plus_lora too, since the "it never wins" conclusion was MME-specific
  - 5 independent random 5,000-question subsets of VQA-v2 per point
    (different --split-seed each), averaged -- gives both a mean estimate
    and a sense of run-to-run spread, without a formal significance test
  - Just eyeball where the averaged curves cross

Usage:
    python threshold_check_vqav2.py --gpus 0,1,2,3
"""
import re
import json
import queue
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path("/home/yenhsiu/AdaptSplit-LMM")
VALIDATE_PY  = PROJECT_ROOT / "validate.py"
MOBILEVLM_PY = "/home/yenhsiu/.pyenv/versions/anaconda3-5.0.0/envs/mobilevlm/bin/python"

N_POINTS   = [50, 150, 250, 350, 450, 550]
MODELS     = ["base_lora", "plus_lora", "original"]
SEEDS      = [42, 43, 44, 45, 46]
SPLIT_RATIO = 0.023326  # ~5000 of VQA-v2's 214,354 questions

LOG_PATH = Path(__file__).parent / "threshold_check_vqav2.log.jsonl"
log_lock = threading.Lock()
gpu_pool = None


class GPUPool:
    def __init__(self, ids):
        self.q = queue.Queue()
        for i in ids:
            self.q.put(i)

    def acquire(self):
        return self.q.get()

    def release(self, gid):
        self.q.put(gid)


def log(entry: dict):
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    with log_lock, LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_one(N: int, model: str, seed: int) -> float:
    gid = gpu_pool.acquire()
    try:
        cmd = [MOBILEVLM_PY, str(VALIDATE_PY),
               "--dataset", "vqav2", "--no-quant", "--n-tokens", str(N),
               "--merge", "--force-model", model,
               "--split", "search", "--split-ratio", str(SPLIT_RATIO),
               "--split-seed", str(seed), "--cuda", str(gid)]
        print(f"[start] cuda{gid} N={N} model={model} seed={seed}", flush=True)
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"[FAIL] N={N} model={model} seed={seed}\n{output[-1500:]}", flush=True)
            raise RuntimeError(f"validate.py failed: N={N} model={model} seed={seed}")
        m = re.search(r"Overall Accuracy:\s*([\d.]+)%", output)
        score = float(m.group(1)) if m else None
        print(f"[done]  cuda{gid} N={N} model={model} seed={seed} -> {score}", flush=True)
    finally:
        gpu_pool.release(gid)
    log({"N": N, "model": model, "seed": seed, "score": score})
    return score


def main():
    global gpu_pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    gpu_pool = GPUPool(gpu_ids)
    LOG_PATH.write_text("")

    jobs = [(N, m, s) for N in N_POINTS for m in MODELS for s in SEEDS]
    print(f"=== {len(jobs)} runs queued across GPUs {gpu_ids} ===", flush=True)

    results = {}  # (N, model) -> [scores]
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = {ex.submit(run_one, N, m, s): (N, m, s) for N, m, s in jobs}
        for fut in futs:
            N, m, s = futs[fut]
            score = fut.result()
            results.setdefault((N, m), []).append(score)

    print("\n=== Averaged results (mean over 5 seeds) ===")
    print(f"{'N':>5} | {'base_lora':>20} | {'plus_lora':>20} | {'original':>20}")
    summary = {}
    for N in N_POINTS:
        row = []
        for m in MODELS:
            scores = [s for s in results[(N, m)] if s is not None]
            mean = sum(scores) / len(scores) if scores else float("nan")
            spread = f"[{min(scores):.2f}-{max(scores):.2f}]" if scores else "n/a"
            row.append(f"{mean:6.2f} {spread}")
            summary.setdefault(str(N), {})[m] = {"mean": mean, "scores": scores}
        print(f"{N:>5} | {row[0]:>20} | {row[1]:>20} | {row[2]:>20}")

    out_path = Path(__file__).parent / "threshold_check_vqav2.result.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
