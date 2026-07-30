"""
validate_sweep.py

For each budget in search result JSONs, run validation on TextVQA and POPE:
  - All B=1  : n4=0, n2=0, n1 = B // 1024
  - All B=2  : n4=0, n2 = B // 2048, n1=0
  - All B=4  : n4 = B // 4096, n2=0, n1=0
  - MME-opt  : best (n4, n2, n1) from search result JSON

Results are appended to:
  results/TextVQA.txt
  results/POPE.txt

Usage:
    python validate_sweep.py --search-results results/search_mme_*.json --cuda 0
    python validate_sweep.py --budgets 25600 115500 266240 --cuda 0
"""

import re
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON       = "python"
VALIDATE_PY  = PROJECT_ROOT / "validate.py"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Score parsers ─────────────────────────────────────────────────────────────

def parse_textvqa(text: str) -> str:
    m = re.search(r"Accuracy:\s*([\d.]+)%", text)
    return f"{float(m.group(1)):.2f}%" if m else "N/A"

def parse_pope(text: str) -> str:
    scores = re.findall(r"F1 score:\s*([\d.]+)", text)
    if not scores:
        return "N/A"
    avg = sum(float(x) for x in scores) / len(scores)
    return f"{avg:.4f}"

def parse_mmbench(text: str) -> str:
    m = re.search(r"MMBench Dev Accuracy:\s*([\d.]+)%", text)
    return f"{float(m.group(1)):.2f}%" if m else "N/A"

def parse_mme(text: str) -> str:
    m = re.search(r"Combined Total Score:\s*([\d.]+)", text)
    return f"{float(m.group(1)):.4f}" if m else "N/A"

PARSERS = {
    "textvqa": parse_textvqa,
    "pope":    parse_pope,
    "mmbench": parse_mmbench,
    "mme":     parse_mme,
}

# ── Validation runner ─────────────────────────────────────────────────────────

def run_validate(dataset: str, n4: int, n2: int, n1: int, cuda: str, merge: bool = False) -> str:
    cmd = [PYTHON, str(VALIDATE_PY),
           "--dataset", dataset,
           "--n4", str(n4), "--n2", str(n2), "--n1", str(n1),
           "--split", "all",
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
        print(f"  [warn] validate failed\n{output[-300:]}")
        return "ERR"
    return PARSERS[dataset](output)


# ── Budget configs ────────────────────────────────────────────────────────────

def uniform_configs(B: int) -> dict:
    return {
        "All B=1": (0,          0,          B // 1024),
        "All B=2": (0,          B // 2048,  0),
        "All B=4": (B // 4096,  0,          0),
    }


# ── Table I/O ─────────────────────────────────────────────────────────────────

COLUMNS = ["B (bits)", "All B=1", "All B=2", "All B=4", "MME-opt"]
SEP     = "\t"

def load_table(path: Path) -> dict:
    """Load existing table as {budget_str: {col: value}}."""
    table = {}
    if not path.exists():
        return table
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return table
    headers = lines[0].split(SEP)
    for line in lines[1:]:
        cells = line.split(SEP)
        if not cells[0].strip():
            continue
        row = dict(zip(headers, cells))
        table[cells[0].strip()] = row
    return table


def save_table(path: Path, table: dict):
    """Write table dict back to file."""
    lines = [SEP.join(COLUMNS)]
    for b_str in sorted(table.keys(), key=lambda x: int(x.replace(",", ""))):
        row = table[b_str]
        lines.append(SEP.join(row.get(c, "") for c in COLUMNS))
    path.write_text("\n".join(lines) + "\n")
    print(f"  [saved] {path}")


def fmt_budget(B: int) -> str:
    return f"{B:,}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search-results", nargs="+", type=Path,
                       help="Path(s) to search result JSON(s)")
    group.add_argument("--budgets", nargs="+", type=int,
                       help="Budget(s) to validate (looks up MME-opt from configs/optimal_configs.json)")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "pope"],
                        choices=["textvqa", "pope", "mmbench", "mme"])
    parser.add_argument("--mme-opt-only", action="store_true",
                        help="Only run MME-opt configs, skip All B=1/2/4 baselines")
    parser.add_argument("--merge", action="store_true",
                        help="Use prune+merge token selection for all runs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save results (default: results/)")
    args = parser.parse_args()

    # ── Build budget → mme_opt mapping ──
    budget_configs = {}  # {B: {"n4": .., "n2": .., "n1": ..}}

    if args.search_results:
        for json_path in args.search_results:
            data = json.loads(json_path.read_text())
            for r in data.get("results", [data]):
                B = r["S"]
                budget_configs[B] = {"n4": r["n4"], "n2": r["n2"], "n1": r["n1"]}
    else:
        # Load from optimal_configs.json
        cfg_path = PROJECT_ROOT / "configs" / "optimal_configs.json"
        if not cfg_path.exists():
            print("[error] configs/optimal_configs.json not found. Run search first.")
            sys.exit(1)
        cfg = json.loads(cfg_path.read_text())
        for B in args.budgets:
            key = str(B)
            if key not in cfg["budgets"]:
                print(f"[warn] Budget {B} not in optimal_configs.json, skipping MME-opt")
                budget_configs[B] = None
            else:
                c = cfg["budgets"][key]
                budget_configs[B] = {"n4": c["n4"], "n2": c["n2"], "n1": c["n1"]}

    out_dir = args.output_dir if args.output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Per dataset ──
    for dataset in args.datasets:
        out_path = out_dir / f"{dataset.upper()}_{timestamp}.txt"
        table    = load_table(out_path)

        for B, opt in sorted(budget_configs.items()):
            b_str = fmt_budget(B)
            if b_str not in table:
                table[b_str] = {"B (bits)": b_str}

            uni = uniform_configs(B)

            # All B=1 / B=2 / B=4
            if not args.mme_opt_only:
                for label, (n4, n2, n1) in uni.items():
                    if table[b_str].get(label):
                        print(f"[skip] {dataset} | {b_str} | {label} (already done)")
                        continue
                    N = n4 + n2 + n1
                    if N == 0:
                        table[b_str][label] = "N/A"
                        continue
                    print(f"[run]  {dataset} | {b_str} | {label}  (n4={n4}, n2={n2}, n1={n1}, N={N})")
                    table[b_str][label] = run_validate(dataset, n4, n2, n1, args.cuda, merge=args.merge)
                    save_table(out_path, table)

            # MME-opt
            if table[b_str].get("MME-opt"):
                print(f"[skip] {dataset} | {b_str} | MME-opt (already done)")
            elif opt is None:
                table[b_str]["MME-opt"] = "N/A"
            else:
                n4, n2, n1 = opt["n4"], opt["n2"], opt["n1"]
                print(f"[run]  {dataset} | {b_str} | MME-opt  (n4={n4}, n2={n2}, n1={n1})")
                table[b_str]["MME-opt"] = run_validate(dataset, n4, n2, n1, args.cuda, merge=args.merge)
                save_table(out_path, table)

        print(f"\n=== {dataset.upper()} ===")
        print(out_path.read_text())


if __name__ == "__main__":
    main()
