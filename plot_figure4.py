"""
Figure 4: Lookup Table interpolation accuracy on VQAv2.

X axis: Budget (continuous, 7 interpolation points added between the 8 existing anchors)
Y axis: VQAv2 Accuracy
Series:
    Oracle          - the 8 existing anchor points (dots only)
    Snap-down       - staircase: holds the next-LOWER anchor's full config/accuracy
    Snap-and-refill - lower anchor's (n4, n2) kept, n1 topped up with the leftover
                      budget (more B=1 tokens) -- smooth curve

Data status (2026-07-28):
    - Oracle: REAL data. All 8 anchor configs already had full 5,000-question
      VQAv2 answer .jsonl files on disk (playground/data/eval/vqav2/answers/...),
      so this was scored locally against
      /mnt/ssd/yenhsiu_datasets/vqav2/v2_mscoco_val2014_annotations.json with
      llava/eval/eval_vqav2.py -- no new inference, no GPU used.
    - Snap-down: derived for free from Oracle (no new inference) -- it's just the
      lower anchor's already-known accuracy, held flat across the gap.
    - Snap-and-refill: needs NEW inference (confirmed missing -- none of the 7
      refill configs below exist yet under playground/data/eval/vqav2/answers/...).
      REFILL_VQAV2 is left as None per point; run `--print-commands` to get the
      exact validate.py invocations, then fill in REFILL_VQAV2 (or point
      REFILL_ANSWER_FILES at the resulting .jsonl) once available.

Supplementary table (numbers only, no plot) per the spec: same 7 interpolation
points x {snap-down, snap-and-refill} evaluated on MME/POPE/TextVQA/MMBench too,
to check whether the refill improvement seen on VQAv2 generalizes.
    - snap-down column: FREE, reuses the lower anchor's already-known score from
      plot_paper_figures.DATA (no new inference -- snap-down never changes the
      inference config, only the "logical" budget label).
    - snap-and-refill column: needs the SAME new inference runs as above, just
      also scored against MME/POPE/TextVQA/MMBench.

Usage:
    python plot_figure4.py                  # draws what's available; warns about the rest
    python plot_figure4.py --print-commands # print validate.py commands for the missing runs
    python plot_figure4.py --rescore        # re-run local VQAv2 scoring for the 8 anchors
"""
import argparse
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_data import BUDGETS, DATA

PROJECT_ROOT = Path(__file__).parent.resolve()
ANSWERS_DIR = PROJECT_ROOT / "playground/data/eval/vqav2/answers/llava_vqav2_mscoco_val2014"
ANNOTATION_FILE = "/mnt/ssd/yenhsiu_datasets/vqav2/v2_mscoco_val2014_annotations.json"

# ── 8 anchors: (n4, n2, n1) of the VQAv2-optimized config at each budget,
#    from results/search_vqav2_*.json / results/vqa_anchore/search_vqav2_*.json ──
ANCHOR_CONFIGS = {
    25600:  (0, 0, 25),
    41000:  (0, 1, 38),
    66000:  (2, 15, 26),
    106000: (0, 35, 33),
    170000: (1, 24, 114),
    273000: (37, 1, 116),
    439000: (0, 36, 356),
    589824: (7, 134, 280),
}

# Answer files already on disk for the 8 anchors (scored below -> ORACLE_VQAV2)
ANCHOR_ANSWER_FILES = {
    25600:  "validate_vqav2_search_n40_n20_n125.jsonl",
    41000:  "validate_vqav2_search_n40_n21_n138.jsonl",
    66000:  "validate_vqav2_search_n42_n215_n126.jsonl",
    106000: "validate_vqav2_search_n40_n235_n133.jsonl",
    170000: "validate_vqav2_search_n41_n224_n1114.jsonl",
    273000: "validate_vqav2_search_n437_n21_n1116.jsonl",
    439000: "validate_vqav2_search_n40_n236_n1356.jsonl",
    589824: "validate_vqav2_search_n47_n2134_n1280.jsonl",
}

# Measured 2026-07-28 by scoring the files above with llava/eval/eval_vqav2.py
# (see ORACLE_VQAV2 population in main() / --rescore to regenerate)
ORACLE_VQAV2 = {
    25600: 67.71, 41000: 69.76, 66000: 70.84, 106000: 71.67,
    170000: 72.29, 273000: 72.96, 439000: 74.62, 589824: 75.11,
}

TOKEN_DIM = 1024


def refill_config(budget_mid: float, lower_anchor: int):
    """Lower anchor's (n4, n2) fixed; n1 topped up with the leftover budget."""
    n4, n2, _ = ANCHOR_CONFIGS[lower_anchor]
    fixed_bits = (4 * n4 + 2 * n2) * TOKEN_DIM
    leftover_bits = budget_mid - fixed_bits
    n1 = max(0, int(leftover_bits // TOKEN_DIM))
    n1 = min(n1, 576 - n4 - n2)  # token-count cap
    return n4, n2, n1


def build_interpolation_points():
    """One geometric-midpoint interpolation point per gap between consecutive anchors."""
    points = []
    for lo, hi in zip(BUDGETS[:-1], BUDGETS[1:]):
        mid = (lo * hi) ** 0.5
        n4, n2, n1 = refill_config(mid, lower_anchor=lo)
        actual_budget = (4 * n4 + 2 * n2 + n1) * TOKEN_DIM
        points.append({
            "budget_target": mid,
            "lower_anchor": lo,
            "upper_anchor": hi,
            "refill_config": (n4, n2, n1),
            "actual_budget": actual_budget,
            "snap_down_acc": ORACLE_VQAV2[lo],  # free: reuse lower anchor's score
        })
    return points


# Fill in once the corresponding validate.py runs + eval_vqav2.py scoring exist.
# Keyed by budget_target (from build_interpolation_points()); value = VQAv2 accuracy (%).
REFILL_VQAV2 = {}

# Same interpolation points, evaluated on the other 4 benchmarks (supplementary
# table). Keyed by budget_target -> {"MME": .., "POPE": .., "TextVQA": .., "MMBench": ..}
REFILL_OTHER_BENCHMARKS = {}


def score_vqav2_jsonl(jsonl_path: Path) -> float:
    result = subprocess.run(
        ["python", "llava/eval/eval_vqav2.py",
         "--annotation-file", ANNOTATION_FILE,
         "--result-file", str(jsonl_path)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Overall Accuracy"):
            return float(line.split(":")[1].strip().rstrip("%)").split("%")[0].strip().rstrip("("))
    raise RuntimeError(f"could not parse eval_vqav2.py output for {jsonl_path}:\n{result.stdout}\n{result.stderr}")


def rescore_anchors():
    print("Re-scoring the 8 anchor VQAv2 answer files (local, no GPU)...")
    for budget, fname in ANCHOR_ANSWER_FILES.items():
        path = ANSWERS_DIR / fname
        acc = score_vqav2_jsonl(path)
        ORACLE_VQAV2[budget] = acc
        print(f"  budget={budget:>7}  {fname}  ->  {acc:.2f}%")


def print_missing_run_commands(points):
    print("\n# ---- snap-and-refill: new inference needed, 7 configs x 5 datasets ----")
    print("# Machine is shared -- check GPU availability before launching (see earlier discussion).")
    for p in points:
        n4, n2, n1 = p["refill_config"]
        tag = f"n4{n4}_n2{n2}_n1{n1}"
        print(f"\n# budget~={p['budget_target']:.0f}  (between anchors {p['lower_anchor']} and {p['upper_anchor']})  config={tag}")
        for dataset in ["vqav2", "mme", "pope", "textvqa", "mmbench"]:
            print(f"bash validate.sh --dataset {dataset} --n4 {n4} --n2 {n2} --n1 {n1} --merge --cuda 0")


def plot_figure4(points, out_path="figure4_lut_interpolation_vqav2.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Oracle: the 8 real anchor points
    oracle_x = sorted(ORACLE_VQAV2)
    oracle_y = [ORACLE_VQAV2[b] for b in oracle_x]
    ax.scatter(oracle_x, oracle_y, color="#e34948", marker='o', s=70, zorder=5,
               label="Oracle (measured anchors)")

    # Snap-down: staircase, holds the lower anchor's accuracy across each gap
    stair_x, stair_y = [], []
    for lo, hi in zip(BUDGETS[:-1], BUDGETS[1:]):
        stair_x += [lo, hi]
        stair_y += [ORACLE_VQAV2[lo], ORACLE_VQAV2[lo]]
    ax.plot(stair_x, stair_y, color="#888780", linewidth=2, linestyle="--",
             drawstyle="steps-post", label="Snap-down (staircase)")

    # Snap-and-refill: only plot measured interpolation points (real data), never
    # invent a "smooth curve" by connecting bare anchors -- that would look like a
    # measured result when it's actually zero new data.
    refill_x = [p["actual_budget"] for p in points if REFILL_VQAV2.get(p["budget_target"]) is not None]
    refill_y = [REFILL_VQAV2[p["budget_target"]] for p in points if REFILL_VQAV2.get(p["budget_target"]) is not None]
    missing = [p["budget_target"] for p in points if REFILL_VQAV2.get(p["budget_target"]) is None]

    if refill_x:
        order = sorted(range(len(refill_x)), key=lambda i: refill_x[i])
        refill_x = [refill_x[i] for i in order]
        refill_y = [refill_y[i] for i in order]
        ax.scatter(refill_x, refill_y, color="#1baf7a", marker='s', s=60, zorder=6,
                   label="Snap-and-refill (measured interpolation points)")
    if missing:
        warnings.warn(
            f"Snap-and-refill has NO measured data yet ({len(missing)}/{len(points)} interpolation "
            f"points missing, budgets ~{[round(m) for m in missing]}) -- new inference not run. "
            f"Not drawing a line for this series (would misrepresent unmeasured data as a smooth "
            f"result). Run --print-commands for the exact jobs, then re-run this script.",
            stacklevel=2,
        )
    else:
        # Only connect with a line once every interpolation point is real data.
        full_x = sorted(oracle_x + refill_x)
        full_y = [dict(zip(oracle_x, oracle_y), **dict(zip(refill_x, refill_y)))[x] for x in full_x]
        ax.plot(full_x, full_y, color="#1baf7a", linewidth=2, alpha=0.7, zorder=4)

    ax.set_xscale('log')
    ax.set_xlabel('Transmission Budget (bits, log scale)')
    ax.set_ylabel('VQAv2 Accuracy (%)')
    ax.set_title('Figure 4: Lookup Table Interpolation Accuracy (VQAv2)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-commands", action="store_true",
                        help="Print validate.py commands for the missing snap-and-refill runs")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-run local VQAv2 scoring for the 8 anchors instead of using cached values")
    args = parser.parse_args()

    if args.rescore:
        rescore_anchors()

    interp_points = build_interpolation_points()

    if args.print_commands:
        print_missing_run_commands(interp_points)
    else:
        plot_figure4(interp_points)
        n_missing = sum(1 for p in interp_points if REFILL_VQAV2.get(p["budget_target"]) is None)
        if n_missing:
            print(f"\n{n_missing}/{len(interp_points)} snap-and-refill points still need new inference.")
            print("Run `python plot_figure4.py --print-commands` for the exact validate.py commands.")
