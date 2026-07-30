"""
Threshold calibration for base_lora / plus_lora / original checkpoint routing.

Method:
  - Pure N effect only: --no-quant --n-tokens N (no quantization confound)
  - --force-model overrides automatic select_model() so any candidate can be
    run at an exact N
  - At each candidate N, all THREE checkpoints are run and compared pairwise
    with McNemar's test (paired, same MME questions) -- no assumption that
    exactly two boundaries exist in a fixed base->plus->original order. If
    plus_lora never comes out on top anywhere, the result correctly collapses
    to a 2-tier (base/original) rule instead of forcing a 3-tier one.
  - Coarse pass at sparse anchors first, then adaptive bisection ONLY between
    adjacent anchors whose top model differs -- not a blind dense grid.
  - floor=16, ceiling=576 (the full feasible token-count range).

Usage:
    python threshold_calibration.py [--gpus 0,1,2,3]
"""
import sys
import json
import queue
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

PROJECT_ROOT = Path("/home/yenhsiu/AdaptSplit-LMM")
VALIDATE_PY  = PROJECT_ROOT / "validate.py"
MOBILEVLM_PY = "/home/yenhsiu/.pyenv/versions/anaconda3-5.0.0/envs/mobilevlm/bin/python"
ANSWERS_DIR  = PROJECT_ROOT / "playground/data/eval/MME/eval_tool/answers"

MODELS = ["base_lora", "plus_lora", "original"]
COARSE_ANCHORS = [16, 32, 144, 288, 432, 576]
FLOOR, CEILING = 16, 576

LOG_PATH = Path(__file__).parent / "threshold_calibration.log.jsonl"
log_lock = threading.Lock()

EVAL_CATEGORIES = [
    "existence", "count", "position", "color", "posters", "celebrity", "scene",
    "landmark", "artwork", "OCR", "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]


class GPUPool:
    def __init__(self, ids):
        self.q = queue.Queue()
        for i in ids:
            self.q.put(i)

    def acquire(self):
        return self.q.get()

    def release(self, gid):
        self.q.put(gid)


gpu_pool = None  # set in main() based on --gpus


def log(entry: dict):
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    with log_lock, LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_pred_ans(pred_ans: str) -> str:
    """Same normalization as calculation.py's parse_pred_ans."""
    pred_ans = pred_ans.strip().lower()
    if pred_ans in ("yes", "no"):
        return pred_ans
    prefix = pred_ans[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def load_per_question(answers_dir: Path) -> dict:
    """Return {(category, file, prompt): (gt_ans, pred_label)}.

    A record is "file\\tprompt\\tgt_ans\\tanswer" written with a trailing
    newline -- but an un-tuned checkpoint (e.g. `original`) can generate an
    answer that itself contains a literal newline, which then looks like a
    second, malformed physical line (no tabs) when read naively. Any physical
    line with fewer than 3 tabs is treated as a continuation of the previous
    record's answer text, not a new record."""
    data = {}
    for cat in EVAL_CATEGORIES:
        fpath = answers_dir / f"{cat}.txt"
        if not fpath.exists():
            continue
        records = []
        for line in fpath.read_text().split("\n"):
            if line.count("\t") >= 3 or not records:
                records.append(line)
            else:
                records[-1] += "\n" + line
        for rec in records:
            if not rec.strip():
                continue
            file_, prompt, gt_ans, pred_ans = rec.split("\t", 3)
            data[(cat, file_, prompt)] = (gt_ans.strip().lower(), parse_pred_ans(pred_ans))
    return data


def compute_mme_score(data: dict) -> float:
    """Recompute the official MME total score from a per-question data dict,
    matching calculation.py's exact logic: per category, (acc + acc_plus)*100,
    summed across all 14 categories. `acc` is per-question accuracy;
    `acc_plus` is the fraction of images where BOTH of its 2 questions are
    correct. Reported alongside `acc` for readability -- McNemar's test still
    runs on the per-question labels themselves, not on this aggregate."""
    from collections import defaultdict
    by_cat = defaultdict(list)
    for (cat, file_, prompt), (gt, pred) in data.items():
        by_cat[cat].append((file_, gt == pred))

    total = 0.0
    for cat, items in by_cat.items():
        acc = sum(ok for _, ok in items) / len(items)

        by_file = defaultdict(list)
        for file_, ok in items:
            by_file[file_].append(ok)
        acc_plus = sum(1 for oks in by_file.values() if len(oks) == 2 and all(oks)) / len(by_file)

        total += (acc + acc_plus) * 100
    return total


# A given (N, model) may be requested by multiple points in the search
# (e.g. base_lora@144 is needed both by the coarse pass and possibly by a
# later bisection). A per-key lock + cache ensures it's only ever actually
# run once; concurrent/later requests reuse the cached per-question data
# instead of racing on the same output file.
_key_locks = {}
_key_locks_guard = threading.Lock()
_data_cache = {}


def _lock_for(key):
    with _key_locks_guard:
        if key not in _key_locks:
            _key_locks[key] = threading.Lock()
        return _key_locks[key]


def _answers_dir_complete(answers_dir: Path) -> bool:
    return answers_dir.is_dir() and all((answers_dir / f"{c}.txt").exists() for c in EVAL_CATEGORIES)


def get_question_data(N: int, model: str) -> dict:
    """Run (or reuse) validate.py for MME, no quant, forced checkpoint.
    Returns {(category, file, prompt): (gt_ans, pred_label)}, cached by (N, model).
    Also reuses a completed run already on disk from a previous invocation of
    this script, so restarts don't redo work."""
    key = (N, model)
    with _lock_for(key):
        if key in _data_cache:
            print(f"[cache] N={N} model={model} (reusing prior run)", flush=True)
            return _data_cache[key]

        exp_name = f"validate_mme_noq_N{N}_{model}"
        existing_dir = ANSWERS_DIR / exp_name
        if _answers_dir_complete(existing_dir):
            print(f"[disk]  N={N} model={model} (found completed run on disk, skipping rerun)", flush=True)
            data = load_per_question(existing_dir)
            _data_cache[key] = data
            return data

        gid = gpu_pool.acquire()
        try:
            cmd = [MOBILEVLM_PY, str(VALIDATE_PY),
                   "--dataset", "mme", "--no-quant", "--n-tokens", str(N),
                   "--merge", "--force-model", model,
                   "--split", "all", "--cuda", str(gid)]
            print(f"[start] cuda{gid} N={N} model={model}", flush=True)
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            output = result.stdout + result.stderr
            if result.returncode != 0:
                print(f"[FAIL] N={N} model={model}\n{output[-1500:]}", flush=True)
                raise RuntimeError(f"validate.py failed: N={N} model={model}")
            print(f"[done]  cuda{gid} N={N} model={model}", flush=True)
        finally:
            gpu_pool.release(gid)

        data = load_per_question(existing_dir)
        _data_cache[key] = data
        return data


def mcnemar_between(N: int, model_a: str, model_b: str, data_a: dict, data_b: dict) -> dict:
    """Pure statistic computation -- no inference, just compares two already-fetched
    per-question datasets."""
    keys = set(data_a) & set(data_b)
    if not keys:
        raise RuntimeError(f"No overlapping questions for N={N} {model_a}/{model_b}")

    b_count = c_count = correct_a = correct_b = 0
    for k in keys:
        gt, pred_a = data_a[k]
        _, pred_b = data_b[k]
        ca = pred_a == gt
        cb = pred_b == gt
        correct_a += ca
        correct_b += cb
        if ca and not cb:
            b_count += 1
        elif cb and not ca:
            c_count += 1

    n = len(keys)
    acc_a, acc_b = correct_a / n, correct_b / n

    table = np.array([[0, b_count], [c_count, 0]])
    exact = (b_count + c_count) < 25
    result = mcnemar(table, exact=exact, correction=True)

    winner = None
    if result.pvalue < 0.05:
        winner = model_a if b_count > c_count else model_b

    out = {
        "N": N, "model_a": model_a, "model_b": model_b,
        "acc_a": acc_a, "acc_b": acc_b,
        "b": b_count, "c": c_count, "n_questions": n,
        "statistic": float(result.statistic), "pvalue": float(result.pvalue),
        "winner": winner,
        "low_power_warning": (b_count + c_count) < 20,
    }
    log(out)
    print(f"  [pair] N={N} | {model_a}={acc_a:.4f} vs {model_b}={acc_b:.4f} | "
          f"b={b_count} c={c_count} p={result.pvalue:.4f} -> "
          f"{'winner=' + winner if winner else 'NOT SIGNIFICANT'}"
          f"{' [LOW POWER]' if out['low_power_warning'] else ''}", flush=True)
    return out


def compare_all_three(N: int) -> dict:
    """Fetch all 3 checkpoints at N, run all 3 pairwise McNemar tests, and
    determine the winning set: whichever model(s) are never significantly
    beaten by another. Also reports the raw-accuracy top model as `top1`."""
    print(f"=== N={N}: fetching all 3 checkpoints ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {m: ex.submit(get_question_data, N, m) for m in MODELS}
        data = {m: f.result() for m, f in futs.items()}

    pairs = [("base_lora", "plus_lora"), ("plus_lora", "original"), ("base_lora", "original")]
    pair_results = {}
    beaten_by = {m: set() for m in MODELS}
    acc = {}
    for a, b in pairs:
        r = mcnemar_between(N, a, b, data[a], data[b])
        pair_results[(a, b)] = r
        acc[a] = r["acc_a"]
        acc[b] = r["acc_b"]
        if r["winner"] == a:
            beaten_by[b].add(a)
        elif r["winner"] == b:
            beaten_by[a].add(b)

    winning_set = [m for m in MODELS if not beaten_by[m]]
    top1 = max(MODELS, key=lambda m: acc[m])
    mme_score = {m: compute_mme_score(data[m]) for m in MODELS}

    out = {
        "N": N, "acc": acc, "mme_score": mme_score, "winning_set": winning_set, "top1": top1,
        "pairs": {f"{a}_vs_{b}": r for (a, b), r in pair_results.items()},
    }
    print(f"=== N={N} result: winning_set={winning_set} top1(acc)={top1} "
          f"| mme_score={ {m: round(v, 1) for m, v in mme_score.items()} } ===\n", flush=True)
    return out


# ── Stage 2: adaptive bisection between coarse anchors whose winner differs ──

def bisect_segment(Na: int, Nb: int, ra: dict, rb: dict, tol: int = 8, max_depth: int = 6, depth: int = 0):
    """ra/rb are compare_all_three() results already computed at Na/Nb, with
    different top1. Recursively narrow down where the change happens."""
    if Nb - Na <= tol or depth >= max_depth:
        return []

    mid = (Na + Nb) // 2
    rmid = compare_all_three(mid)
    results = [rmid]

    if rmid["top1"] == ra["top1"]:
        results += bisect_segment(mid, Nb, rmid, rb, tol, max_depth, depth + 1)
    elif rmid["top1"] == rb["top1"]:
        results += bisect_segment(Na, mid, ra, rmid, tol, max_depth, depth + 1)
    else:
        # a third model takes over in the middle -- recurse both halves
        results += bisect_segment(Na, mid, ra, rmid, tol, max_depth, depth + 1)
        results += bisect_segment(mid, Nb, rmid, rb, tol, max_depth, depth + 1)
    return results


def main():
    global gpu_pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3",
                        help="Comma-separated CUDA device ids to use (default: 0,1,2,3)")
    args = parser.parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    gpu_pool = GPUPool(gpu_ids)
    print(f"=== Using GPUs: {gpu_ids} | anchors: {COARSE_ANCHORS} ===", flush=True)

    # ── Stage 1: coarse pass ──
    coarse_results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {N: ex.submit(compare_all_three, N) for N in COARSE_ANCHORS}
        for N, f in futs.items():
            coarse_results[N] = f.result()

    print("\n=== Coarse pass summary ===")
    for N in COARSE_ANCHORS:
        r = coarse_results[N]
        print(f"  N={N:>4}: top1={r['top1']:<10} winning_set={r['winning_set']} "
              f"acc={r['acc']} mme_score={r['mme_score']}")

    # ── Stage 2: adaptive bisection only where adjacent anchors disagree ──
    all_results = dict(coarse_results)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        for Na, Nb in zip(COARSE_ANCHORS, COARSE_ANCHORS[1:]):
            ra, rb = coarse_results[Na], coarse_results[Nb]
            if ra["top1"] != rb["top1"]:
                print(f"\n=== Transition detected between N={Na} ({ra['top1']}) "
                      f"and N={Nb} ({rb['top1']}) -- bisecting ===", flush=True)
                futs.append(ex.submit(bisect_segment, Na, Nb, ra, rb))
            else:
                print(f"\n=== N={Na}..{Nb}: same top1 ({ra['top1']}), no bisection needed ===", flush=True)
        for f in futs:
            for r in f.result():
                all_results[r["N"]] = r

    # ── Final N -> model table ──
    print("\n=== FINAL N -> MODEL TABLE ===")
    for N in sorted(all_results):
        r = all_results[N]
        print(f"  N={N:>4}: top1={r['top1']:<10} winning_set={r['winning_set']} "
              f"mme_score={r['mme_score']}")

    out_path = Path(__file__).parent / "threshold_calibration.result.json"
    out_path.write_text(json.dumps(
        {str(N): r for N, r in sorted(all_results.items())}, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()