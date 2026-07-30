"""
quantization_search.py

Optuna ask-and-tell search for optimal (n4, n2, n1) token allocation.
Each trial runs evaluation via validate.py, then reports the score back.

Constraints:
  4·n4 + 2·n2 + n1 = B // 1024   (equality: always use full budget)
  n4 + n2 + n1 ≤ 576              (token cap)
  n4, n2, n1 ≥ 0

Sampling: (n4, n2, n1) is parameterized via two cut points (u1, u2) on the
*budget-proportion* simplex (not the raw token-count simplex -- token counts
have unequal per-unit cost: 4/2/1 -- cutting the token-count simplex directly
silently breaks the budget equality for any non-corner point). This makes the
three uniform-bit-width corners (All B=1/2/4) and interior points reachable
with comparable probability; the old n4-then-n2-then-n1 sequential sampling
gave n4 a disproportionately large range, making corner solutions (all three
uniform-bit-width baselines) exponentially unlikely to ever be sampled --
that bias is the confirmed root cause of searched configs underperforming
the trivial uniform baselines on generalization (POPE/MMBench).

Each budget also gets 3-4 anchor points (All B=1, All B=2, All B=4, and --
when the budget supports averaging >=1 bit/token across all 576 tokens --
the "N=576, no pruning" mixed-bit point) explicitly enqueued as the first
trials, so the search can never do worse than a baseline it never tried.

n1 is always the residual that exactly absorbs the budget equation:
n1 = (B // 1024) - 4·n4 - 2·n2

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

PYTHON      = "python"
PROJECT_ROOT = Path(__file__).parent.resolve()



# ── Evaluation via validate.py subprocess ────────────────────────────────────

def run_eval(dataset: str, n4: int, n2: int, n1: int, exp_name: str, cuda: str,
             split: str = "search", split_ratio: float = 0.3, merge: bool = False) -> float:
    """Run validate.py for (n4, n2, n1) and parse the scalar score."""
    cmd = [PYTHON, str(VALIDATE_PY),
           "--dataset", dataset,
           "--n4", str(n4), "--n2", str(n2), "--n1", str(n1),
           "--split", split, "--split-ratio", str(split_ratio),
           "--cuda", cuda]
    if merge:
        cmd.append("--merge")
    result = subprocess.run(
        cmd,
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

N_MAX = 576  # CLIP patch count -- hard cap on n4+n2+n1


def sample_params(trial: optuna.Trial, S: int):
    """Two cut points on the *budget-proportion* simplex (p4, p2, p1), each
    proportion then divided by its tier's per-token cost (4, 2, 1) to get a
    token count. n1 absorbs rounding residual so 4n4+2n2+n1==T always holds
    exactly, regardless of where (u1, u2) land -- including all three
    uniform-bit-width corners."""
    T = S // 1024
    u1 = trial.suggest_float("u1", 0, 1)
    u2 = trial.suggest_float("u2", 0, 1)
    lo, hi = min(u1, u2), max(u1, u2)
    p4, p2 = lo, hi - lo  # p1 = 1 - hi, implicit

    n4 = round(p4 * T / 4)
    n2 = round(p2 * T / 2)
    n1 = T - 4 * n4 - 2 * n2
    return n4, n2, n1


def is_feasible(n4: int, n2: int, n1: int) -> bool:
    N = n4 + n2 + n1
    return n1 >= 0 and N > 0 and N <= N_MAX


def get_anchor_points(S: int) -> list:
    """All-B1 / All-B2 / All-B4, plus (when the budget allows an average of
    at least 1 bit/token across all N_MAX tokens) the "no pruning" point --
    each only included if it's actually feasible at this exact budget."""
    T = S // 1024
    anchors = []

    if T <= N_MAX:  # else n1=T > N_MAX tokens needed: infeasible, must skip
        anchors.append({"n4": 0, "n2": 0, "n1": T})
    if T <= 2 * N_MAX:
        anchors.append({"n4": 0, "n2": T // 2, "n1": T - 2 * (T // 2)})
    if T <= 4 * N_MAX:
        anchors.append({"n4": T // 4, "n2": 0, "n1": T - 4 * (T // 4)})

    # N=576 (no pruning), 1-bit/2-bit mix -- only exact for T in [N_MAX, 2*N_MAX]
    if N_MAX <= T <= 2 * N_MAX:
        n2 = T - N_MAX
        n1 = N_MAX - n2
        anchors.append({"n4": 0, "n2": n2, "n1": n1})
    # NOTE: for T > 2*N_MAX, the analogous "N=576, 2-bit/4-bit mix" point
    # would need a 3-way solver (not yet implemented) -- deliberately
    # skipped rather than emitting a point that silently underspends budget.

    feasible = [a for a in anchors if is_feasible(a["n4"], a["n2"], a["n1"])]
    deduped = list({(a["n4"], a["n2"], a["n1"]): a for a in feasible}.values())
    return deduped


def anchor_to_u1u2(n4: int, n2: int, n1: int, T: int) -> dict:
    """Inverse of sample_params: given a target (n4, n2, n1), find (u1, u2)
    that reproduce it exactly (up to the same rounding sample_params itself
    would apply)."""
    p4 = 4 * n4 / T
    p2 = 2 * n2 / T
    lo, hi = p4, p4 + p2
    return {"u1": lo, "u2": hi}



def search(dataset: str, S: int, n_trials: int, cuda: str, split: str = "all",
           split_ratio: float = 0.3, merge: bool = False, seed: int = 42) -> dict:
    T = S // 1024

    # Always a fresh study -- (u1, u2) is a different parameter space than
    # the old (n4, n2) sampler, so there is nothing meaningful to resume.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    anchors = get_anchor_points(S)
    anchor_keys = {(a["n4"], a["n2"], a["n1"]) for a in anchors}
    for a in anchors:
        study.enqueue_trial(anchor_to_u1u2(a["n4"], a["n2"], a["n1"], T))
    print(f"  [anchors] {len(anchors)} enqueued: {anchors}")

    history = []  # full per-trial record, incl. pruned
    trial_count = 0
    while trial_count < n_trials:
        trial = study.ask()
        n4, n2, n1 = sample_params(trial, S)
        is_anchor = (n4, n2, n1) in anchor_keys

        if not is_feasible(n4, n2, n1):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            history.append({"trial": trial_count, "u1": trial.params["u1"], "u2": trial.params["u2"],
                             "n4": n4, "n2": n2, "n1": n1, "N": n4 + n2 + n1,
                             "source": "anchor" if is_anchor else "tpe",
                             "score": None, "state": "pruned"})
            continue

        source = "anchor" if is_anchor else "tpe"
        exp_name = f"search_{dataset}_S{S}_t{trial_count:03d}_n4{n4}_n2{n2}_n1{n1}"
        print(f"\n  trial {trial_count:3d} [{source}]: n4={n4:3d}, n2={n2:3d}, n1={n1:3d}, "
              f"N={n4+n2+n1:3d}  →  {exp_name}")

        score = run_eval(dataset, n4, n2, n1, exp_name, cuda, split=split, split_ratio=split_ratio, merge=merge)
        print(f"           score = {score:.4f}")

        study.tell(trial, score)
        history.append({"trial": trial_count, "u1": trial.params["u1"], "u2": trial.params["u2"],
                         "n4": n4, "n2": n2, "n1": n1, "N": n4 + n2 + n1,
                         "source": source, "score": score, "state": "complete"})
        trial_count += 1

    # ── Summary (from our own history, not study.best_trial -- avoids
    #    re-deriving n4/n2/n1 from raw u1/u2 params) ──
    completed = [h for h in history if h["state"] == "complete"]
    best = max(completed, key=lambda h: h["score"])
    n4, n2, n1, N = best["n4"], best["n2"], best["n1"], best["N"]

    print(f"\n[Best] n4={n4}, n2={n2}, n1={n1}, N={N}, source={best['source']}, "
          f"top_ratio={n4/N:.3f}, score={best['score']:.4f}")
    print("[Top 5]")
    for h in sorted(completed, key=lambda h: h["score"], reverse=True)[:5]:
        print(f"  n4={h['n4']:3d}, n2={h['n2']:3d}, n1={h['n1']:3d}, N={h['N']:3d}, "
              f"source={h['source']:>6}, top_ratio={h['n4']/h['N']:.3f}, score={h['score']:.4f}")

    return {"dataset": dataset, "S": S, "n4": n4, "n2": n2, "n1": n1, "N": N,
            "top_ratio": n4 / N if N > 0 else 0.0, "score": best["score"],
            "anchors": anchors, "trials": history}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="mme",
                        choices=["mme", "textvqa", "pope", "scienceqa", "mmbench", "vqav2"])
    parser.add_argument("--budgets",     nargs="+", type=int, default=[115500, 266240, 589824])
    parser.add_argument("--n-trials",   type=int,   default=50)
    parser.add_argument("--split",       default="all", choices=["all", "search", "val"],
                        help="Which subset of the dataset each trial evaluates on (default: all)")
    parser.add_argument("--split-ratio", type=float, default=0.3,
                        help="Fraction of data used as search set when --split=search/val (default: 0.3)")
    parser.add_argument("--cuda",        default="0")
    parser.add_argument("--output",      default=None,
                        help="Path to save results JSON (default: results/search_<dataset>_<timestamp>.json)")
    parser.add_argument("--merge", action="store_true",
                        help="Use prune+merge token selection instead of prune-only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Optuna TPE sampler (default: 42)")
    args = parser.parse_args()

    results = []
    for S in args.budgets:
        print(f"\n{'='*40}\nDataset: {args.dataset}  Budget: {S}\n{'='*40}")
        results.append(search(args.dataset, S, args.n_trials, args.cuda, split=args.split,
                               split_ratio=args.split_ratio, merge=args.merge, seed=args.seed))

    print("\n\n=== Search Summary ===")
    import pandas as pd
    summary_only = [{k: v for k, v in r.items() if k not in ("anchors", "trials")} for r in results]
    print(pd.DataFrame(summary_only).set_index(["dataset", "S"]).to_string())

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
