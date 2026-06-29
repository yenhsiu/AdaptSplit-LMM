"""
quantization_search.py

Optuna ask-and-tell search for optimal (n4, n2, n1) token allocation.
Each trial runs evaluation via validate.py, then reports the score back.

Constraints:
  4·n4 + 2·n2 + n1 = B // 1024   (equality: always use full budget)
  n4 + n2 + n1 ≤ 576              (token cap)
  n4, n2, n1 ≥ 0

n1 is derived automatically: n1 = (B // 1024) - 4·n4 - 2·n2

Supported datasets: mme, textvqa, pope, scienceqa, mmbench, vqav2

Usage:
    python quantization_search.py \
        --dataset mme \
        --budget 115500 \
        --n-trials 50 \
        --cuda 0
"""

import os
import re
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import optuna
VALIDATE_PY = Path(__file__).parent.resolve() / "validate.py"

# Score parsers per dataset — each returns a single float from validate.py stdout
SCORE_PARSERS = {
    "mme":        lambda s: float(re.search(r"Combined Total Score:\s*([\d.]+)", s).group(1))
                            if re.search(r"Combined Total Score:\s*([\d.]+)", s) else 0.0,
    "textvqa":    lambda s: float(re.search(r"Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"Accuracy:\s*([\d.]+)%", s) else 0.0,
    "pope":       lambda s: sum(float(x) for x in re.findall(r"F1 score:\s*([\d.]+)", s)) /
                            max(len(re.findall(r"F1 score:\s*([\d.]+)", s)), 1),
    "scienceqa":  lambda s: float(re.search(r"IMG-Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"IMG-Accuracy:\s*([\d.]+)%", s) else 0.0,
    "mmbench":    lambda s: float(re.search(r"MMBench Dev Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"MMBench Dev Accuracy:\s*([\d.]+)%", s) else 0.0,
    "vqav2":      lambda s: float(re.search(r"Overall Accuracy:\s*([\d.]+)%", s).group(1))
                            if re.search(r"Overall Accuracy:\s*([\d.]+)%", s) else 0.0,
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────

PYTHON      = "/mnt/ssd/yenhsiu_envs/llava_eval/bin/python"
MODEL_PATH  = "/mnt/ssd/yuzhang_models/llava-v1.5-7b"
PROJECT_ROOT = Path(__file__).parent.resolve()
MME_DIR     = PROJECT_ROOT / "playground/data/eval/MME"



# ── Evaluation via validate.py subprocess ────────────────────────────────────

def run_eval(dataset: str, n4: int, n2: int, n1: int, exp_name: str, cuda: str,
             split: str = "search", split_ratio: float = 0.3) -> float:
    """Run validate.py for (n4, n2, n1) and parse the scalar score."""
    result = subprocess.run(
        [PYTHON, str(VALIDATE_PY),
         "--dataset", dataset,
         "--n4", str(n4), "--n2", str(n2), "--n1", str(n1),
         "--split", split, "--split-ratio", str(split_ratio),
         "--cuda", cuda,
         "--no-save"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"  [warn] validate.py failed for {exp_name}\n{output[-300:]}")
        return 0.0

    parser = SCORE_PARSERS.get(dataset)
    if parser is None:
        print(f"  [warn] no score parser for dataset={dataset}")
        return 0.0

    try:
        score = parser(output)
    except Exception as e:
        print(f"  [warn] score parse error: {e}\n{output[-300:]}")
        return 0.0

    if score == 0.0:
        print(f"  [warn] parsed score=0.0, raw output:\n{output[-300:]}")
    return score


# ── Optuna ask-and-tell ───────────────────────────────────────────────────────


def sample_params(trial: optuna.Trial, S: int):
    T = S // 1024
    n4 = trial.suggest_int("n4", 0, T // 4)
    n2 = trial.suggest_int("n2", 0, (T - 4 * n4) // 2)
    n1 = T - 4 * n4 - 2 * n2  # equality constraint: always use full budget
    return n4, n2, n1


def is_feasible(n4: int, n2: int, n1: int) -> bool:
    N = n4 + n2 + n1
    return n1 >= 0 and N > 0 and N <= 576


def search(dataset: str, S: int, n_trials: int, cuda: str, split_ratio: float = 0.3) -> dict:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    trial_count = 0
    while trial_count < n_trials:
        trial = study.ask()
        n4, n2, n1 = sample_params(trial, S)

        if not is_feasible(n4, n2, n1):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue

        exp_name = f"search_{dataset}_S{S}_t{trial_count:03d}_n4{n4}_n2{n2}_n1{n1}"
        print(f"\n  trial {trial_count:3d}: n4={n4:3d}, n2={n2:3d}, n1={n1:3d}, "
              f"N={n4+n2+n1:3d}  →  {exp_name}")

        score = run_eval(dataset, n4, n2, n1, exp_name, cuda, split="all", split_ratio=split_ratio)
        print(f"           score = {score:.4f}")

        study.tell(trial, score)
        trial_count += 1

    # ── Summary ──
    T = S // 1024
    best = study.best_trial
    n4 = best.params["n4"]; n2 = best.params["n2"]
    n1 = T - 4 * n4 - 2 * n2
    N  = n4 + n2 + n1

    print(f"\n[Best] n4={n4}, n2={n2}, n1={n1}, N={N}, "
          f"top_ratio={n4/N:.3f}, score={best.value:.4f}")
    print("[Top 5]")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]:
        _n4 = t.params["n4"]; _n2 = t.params["n2"]
        _n1 = T - 4 * _n4 - 2 * _n2
        _N  = _n4 + _n2 + _n1
        if _N == 0: continue
        print(f"  n4={_n4:3d}, n2={_n2:3d}, n1={_n1:3d}, N={_N:3d}, "
              f"top_ratio={_n4/_N:.3f}, score={t.value:.4f}")

    return {"dataset": dataset, "S": S, "n4": n4, "n2": n2, "n1": n1, "N": N,
            "top_ratio": n4/N if N > 0 else 0.0, "score": best.value}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="mme",
                        choices=["mme", "textvqa", "pope", "scienceqa", "mmbench", "vqav2"])
    parser.add_argument("--budgets",     nargs="+", type=int, default=[115500, 266240, 589824])
    parser.add_argument("--n-trials",   type=int,   default=50)
    parser.add_argument("--split-ratio", type=float, default=0.3,
                        help="Fraction of data used as search set (default: 0.3)")
    parser.add_argument("--cuda",        default="0")
    parser.add_argument("--output",      default=None,
                        help="Path to save results JSON (default: results/search_<dataset>_<timestamp>.json)")
    args = parser.parse_args()

    results = []
    for S in args.budgets:
        print(f"\n{'='*40}\nDataset: {args.dataset}  Budget: {S}\n{'='*40}")
        results.append(search(args.dataset, S, args.n_trials, args.cuda, args.split_ratio))

    print("\n\n=== Search Summary ===")
    import pandas as pd
    print(pd.DataFrame(results).set_index(["dataset", "S"]).to_string())

    # ── Save results ──
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        budgets_tag = "_".join(str(S) for S in args.budgets)
        out_path = out_dir / f"search_{args.dataset}_{budgets_tag}.json"

    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "split_ratio": args.split_ratio,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }, indent=2))
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
