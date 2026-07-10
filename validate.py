"""
validate.py

Run inference + scoring for a given dataset and quantization config.
Called by validate.sh — do not invoke directly.

Modes:
  --baseline              : 576 tokens, no quant (original model)
  --no-quant [--n-tokens N]: N tokens (or 576), no quant
  --budget B              : lookup optimal (n4, n2, n1) from configs/optimal_configs.json
  --n4 X --n2 Y --n1 Z   : manual quantization config

Output:
  results/validate_{dataset}_{exp_name}.txt
"""

import os
import re
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

SPLIT_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON       = "/mnt/ssd/yenhsiu_envs/llava_eval/bin/python"
MODEL_PATH        = "/mnt/ssd/yuzhang_models/llava-v1.5-7b"
LORA_MODEL_PATH   = "/mnt/ssd/yuzhang_models/llava-prumerge-vicuna-7b-v1.5-lora"
LORA_MODEL_BASE   = "lmsys/vicuna-7b-v1.5"
CONFIGS_PATH = PROJECT_ROOT / "configs" / "optimal_configs.json"
RESULTS_DIR  = PROJECT_ROOT / "results"

DATASET_CONFIG = {
    "mme": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/MME/llava_mme.jsonl",
        "image_folder":     "playground/data/eval/MME/MME_Benchmark_release_version",
        "answers_dir":      "playground/data/eval/MME/answers",
        "scoring":          "mme",
    },
    "textvqa": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
        "image_folder":     "playground/data/eval/textvqa/train_images",
        "answers_dir":      "playground/data/eval/textvqa/answers",
        "annotation":       "playground/data/eval/textvqa/TextVQA_0.5.1_val.json",
        "scoring":          "textvqa",
    },
    "pope": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/pope/llava_pope_test.jsonl",
        "image_folder":     "/mnt/ssd/yenhsiu_datasets/POPE/coco_val2014",
        "answers_dir":      "playground/data/eval/pope/answers",
        "annotation_dir":   "/mnt/ssd/yenhsiu_datasets/POPE/pope_annotations",
        "scoring":          "pope",
    },
    "scienceqa": {
        "inference_module": "llava.eval.model_vqa_science",
        "question_file":    "playground/data/eval/scienceqa/llava_test_CQM-A.json",
        "image_folder":     "playground/data/eval/scienceqa/images/test",
        "answers_dir":      "playground/data/eval/scienceqa/answers",
        "base_dir":         "playground/data/eval/scienceqa",
        "scoring":          "scienceqa",
        "extra_infer_args": ["--single-pred-prompt"],
    },
    "mmbench": {
        "inference_module": "llava.eval.model_vqa_mmbench",
        "question_file":    "playground/data/eval/mmbench/mmbench_dev_20230712.tsv",
        "answers_dir":      "playground/data/eval/mmbench/answers/mmbench_dev_20230712",
        "annotation_file":  "playground/data/eval/mmbench/mmbench_dev_20230712.tsv",
        "scoring":          "mmbench",
        "extra_infer_args": ["--single-pred-prompt"],
    },
    "vqav2": {
        "inference_module": "llava.eval.model_vqa_loader",
        "question_file":    "playground/data/eval/vqav2/llava_vqav2_mscoco_val2014.jsonl",
        "image_folder":     "/mnt/ssd/yenhsiu_datasets/POPE/coco_val2014",
        "answers_dir":      "playground/data/eval/vqav2/answers/llava_vqav2_mscoco_val2014",
        "annotation":       "/mnt/ssd/yenhsiu_datasets/vqav2/v2_mscoco_val2014_annotations.json",
        "scoring":          "vqav2",
    },
}

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True,
                        choices=["mme", "textvqa", "pope", "scienceqa", "mmbench", "vqav2"])
    parser.add_argument("--cuda",        default="0")
    parser.add_argument("--split",       default="all", choices=["all", "search", "val"],
                        help="Use a subset: 'search' = first split-ratio%%, 'val' = remaining, 'all' = full dataset")
    parser.add_argument("--split-ratio", type=float, default=0.3,
                        help="Fraction of data used as search set (default: 0.3)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to txt file (used during search)")

    parser.add_argument("--merge", action="store_true",
                        help="Use prumerge_quant_merge (prune + merge) instead of prune-only")

    # Mode flags
    parser.add_argument("--baseline",  action="store_true")
    parser.add_argument("--no-quant",  action="store_true")
    parser.add_argument("--n-tokens",  type=int, default=None)
    parser.add_argument("--budget",    type=int, default=None)
    parser.add_argument("--n4",        type=int, default=None)
    parser.add_argument("--n2",        type=int, default=None)
    parser.add_argument("--n1",        type=int, default=None)

    args = parser.parse_args()

    # Validate mode
    modes = [args.baseline, args.no_quant, args.budget is not None,
             any(x is not None for x in [args.n4, args.n2, args.n1])]
    if sum(modes) > 1:
        parser.error("Specify exactly one mode: --baseline / --no-quant / --budget / --n4 --n2 --n1")
    if sum(modes) == 0:
        parser.error("Specify a mode: --baseline / --no-quant [--n-tokens N] / --budget B / --n4 X --n2 Y --n1 Z")

    # Manual quant: all three required
    if any(x is not None for x in [args.n4, args.n2, args.n1]):
        if not all(x is not None for x in [args.n4, args.n2, args.n1]):
            parser.error("--n4, --n2, --n1 must all be specified together")

    return args


def resolve_config(args):
    """Return (n4, n2, n1, use_quant, exp_name) from parsed args."""
    if args.baseline:
        return 0, 0, 576, False, "baseline"

    if args.no_quant:
        n = args.n_tokens if args.n_tokens is not None else 576
        return 0, 0, n, False, f"noq_N{n}"

    if args.budget is not None:
        configs = json.loads(CONFIGS_PATH.read_text())
        key = str(args.budget)
        if key not in configs["budgets"]:
            available = ", ".join(configs["budgets"].keys())
            print(f"[error] Budget {args.budget} not in lookup table. Available: {available}")
            sys.exit(1)
        c = configs["budgets"][key]
        return c["n4"], c["n2"], c["n1"], True, f"budget{args.budget}"

    # Manual
    return args.n4, args.n2, args.n1, True, f"n4{args.n4}_n2{args.n2}_n1{args.n1}"


# ── Split logic ───────────────────────────────────────────────────────────────

def get_split_question_file(dataset_cfg: dict, split: str, ratio: float) -> Path:
    """Return path to split question file, creating it if it doesn't exist yet."""
    question_file = PROJECT_ROOT / dataset_cfg["question_file"]
    if split == "all":
        return question_file

    split_dir = question_file.parent / "splits"
    split_dir.mkdir(exist_ok=True)

    ratio_tag  = f"r{int(ratio * 100)}"
    split_file = split_dir / f"{question_file.stem}_{split}_{ratio_tag}_seed{SPLIT_SEED}{question_file.suffix}"

    if split_file.exists():
        return split_file

    suffix = question_file.suffix

    if suffix == ".jsonl":
        lines = question_file.read_text().strip().splitlines()
        rng = random.Random(SPLIT_SEED)
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        n_search = int(len(indices) * ratio)
        chosen   = sorted(indices[:n_search] if split == "search" else indices[n_search:])
        selected = [lines[i] for i in chosen]
        split_file.write_text("\n".join(selected) + "\n")
        print(f"[split] {split} ({len(selected)}/{len(lines)} items) → {split_file.name}")

    elif suffix == ".json":
        data = json.loads(question_file.read_text())
        if isinstance(data, dict):
            keys = list(data.keys())
            rng  = random.Random(SPLIT_SEED)
            rng.shuffle(keys)
            n_search     = int(len(keys) * ratio)
            chosen_keys  = keys[:n_search] if split == "search" else keys[n_search:]
            selected_data = {k: data[k] for k in chosen_keys}
            split_file.write_text(json.dumps(selected_data, indent=2))
            print(f"[split] {split} ({len(chosen_keys)}/{len(keys)} items) → {split_file.name}")
        else:
            rng = random.Random(SPLIT_SEED)
            shuffled = data[:]
            rng.shuffle(shuffled)
            n_search     = int(len(shuffled) * ratio)
            selected_data = shuffled[:n_search] if split == "search" else shuffled[n_search:]
            split_file.write_text(json.dumps(selected_data, indent=2))
            print(f"[split] {split} ({len(selected_data)}/{len(data)} items) → {split_file.name}")

    elif suffix == ".tsv":
        lines  = question_file.read_text().splitlines()
        header = lines[0]
        rows   = lines[1:]
        rng    = random.Random(SPLIT_SEED)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        n_search = int(len(indices) * ratio)
        chosen   = sorted(indices[:n_search] if split == "search" else indices[n_search:])
        selected = [rows[i] for i in chosen]
        split_file.write_text(header + "\n" + "\n".join(selected) + "\n")
        print(f"[split] {split} ({len(selected)}/{len(rows)} items) → {split_file.name}")

    else:
        raise ValueError(f"Unsupported question file format: {suffix}")

    return split_file


# ── Env builder ───────────────────────────────────────────────────────────────

def build_env(n4: int, n2: int, n1: int, use_quant: bool, cuda: str, merge: bool = False) -> dict:
    N = n4 + n2 + n1
    token_method = "prumerge_quant_merge" if merge else "prumerge_quant"

    if not use_quant:
        return {
            **os.environ,
            "CUDA_VISIBLE_DEVICES":  cuda,
            "HF_HOME":               "/mnt/ssd/yenhsiu_hf_cache",
            "LLAVA_TOKEN_METHOD":    token_method,
            "LLAVA_USE_QUANT":       "false",
            "LLAVA_N_TOKENS":        str(N),
        }

    active = [(n, b) for n, b in [(n4, 4), (n2, 2), (n1, 1)] if n > 0]
    if len(active) == 1:
        quant_mode   = "uniform"
        quant_bits   = str(active[0][1])
        strat_groups = ""
    else:
        quant_mode   = "stratified"
        quant_bits   = ""
        strat_groups = ",".join(f"{n}:{b}" for n, b in [(n4, 4), (n2, 2), (n1, 1)] if n > 0)

    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES":  cuda,
        "HF_HOME":               "/mnt/ssd/yenhsiu_hf_cache",
        "LLAVA_TOKEN_METHOD":    token_method,
        "LLAVA_USE_QUANT":       "true",
        "LLAVA_QUANT_MODE":      quant_mode,
        "LLAVA_N_TOKENS":        str(N),
        "LLAVA_QUANT_BITS":      quant_bits,
        "LLAVA_STRAT_GROUPS":    strat_groups,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_mme(answers_file: Path, exp_name: str, env: dict) -> str:
    mme_dir = PROJECT_ROOT / "playground/data/eval/MME"
    subprocess.run(
        [PYTHON, "convert_answer_to_mme.py", "--experiment", exp_name],
        env=env, cwd=str(mme_dir), check=True
    )
    result = subprocess.run(
        [PYTHON, "calculation.py", "--results_dir", f"answers/{exp_name}"],
        env=env, cwd=str(mme_dir / "eval_tool"),
        capture_output=True, text=True
    )
    totals = re.findall(r"total score:\s*([\d.]+)", result.stdout)
    total = sum(float(x) for x in totals) if totals else 0.0
    summary = result.stdout + f"\n=== Combined Total Score: {total:.4f} ===\n"
    return summary


def score_textvqa(answers_file: Path, dataset_cfg: dict) -> str:
    result = subprocess.run(
        [PYTHON, "-m", "llava.eval.eval_textvqa",
         "--annotation-file", str(PROJECT_ROOT / dataset_cfg["annotation"]),
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_pope(answers_file: Path, dataset_cfg: dict, question_file: Path) -> str:
    result = subprocess.run(
        [PYTHON, "llava/eval/eval_pope.py",
         "--annotation-dir",  dataset_cfg["annotation_dir"],
         "--question-file",   str(question_file),
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_scienceqa(answers_file: Path, dataset_cfg: dict) -> str:
    base_dir = str(PROJECT_ROOT / dataset_cfg["base_dir"])
    stem = answers_file.stem
    result = subprocess.run(
        [PYTHON, "llava/eval/eval_science_qa.py",
         "--base-dir",      base_dir,
         "--result-file",   str(answers_file),
         "--output-file",   str(answers_file.parent / f"{stem}_output.jsonl"),
         "--output-result", str(answers_file.parent / f"{stem}_result.json")],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_mmbench(answers_file: Path, dataset_cfg: dict) -> str:
    result = subprocess.run(
        [PYTHON, "scripts/eval_mmbench_local.py",
         "--annotation-file", str(PROJECT_ROOT / dataset_cfg["annotation_file"]),
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def score_vqav2(answers_file: Path, dataset_cfg: dict) -> str:
    result = subprocess.run(
        [PYTHON, "llava/eval/eval_vqav2.py",
         "--annotation-file", dataset_cfg["annotation"],
         "--result-file",     str(answers_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    n4, n2, n1, use_quant, mode_tag = resolve_config(args)
    N = n4 + n2 + n1

    dataset_cfg  = DATASET_CONFIG[args.dataset]
    split_tag    = f"_{args.split}" if args.split != "all" else ""
    exp_name     = f"validate_{args.dataset}{split_tag}_{mode_tag}"
    answers_file = PROJECT_ROOT / dataset_cfg["answers_dir"] / f"{exp_name}.jsonl"
    env          = build_env(n4, n2, n1, use_quant, args.cuda, merge=args.merge)

    question_file = get_split_question_file(dataset_cfg, args.split, args.split_ratio)

    print(f"=== Validate: {args.dataset.upper()} ===")
    print(f"Mode:    {'baseline' if not use_quant and N==576 else ('no-quant' if not use_quant else 'quant')}")
    print(f"Config:  n4={n4}, n2={n2}, n1={n1}, N={N}")
    print(f"Split:   {args.split} (ratio={args.split_ratio}, seed={SPLIT_SEED})")
    print(f"Exp:     {exp_name}")
    print("=" * 35)

    # ── Inference ──
    inference_module = dataset_cfg.get("inference_module", "llava.eval.model_vqa_loader")
    if args.merge:
        model_args = ["--model-path", LORA_MODEL_PATH, "--model-base", LORA_MODEL_BASE]
    else:
        model_args = ["--model-path", MODEL_PATH]

    cmd = [PYTHON, "-m", inference_module,
           *model_args,
           "--question-file", str(question_file),
           "--answers-file",  str(answers_file),
           "--temperature",   "0",
           "--conv-mode",     "vicuna_v1",
    ]
    if "image_folder" in dataset_cfg:
        cmd += ["--image-folder", dataset_cfg["image_folder"]]
    cmd += dataset_cfg.get("extra_infer_args", [])

    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0 or not answers_file.exists():
        print("[error] Inference failed.")
        sys.exit(1)

    # ── Scoring ──
    scoring = dataset_cfg["scoring"]
    if scoring == "mme":
        output = score_mme(answers_file, exp_name, env)
    elif scoring == "textvqa":
        output = score_textvqa(answers_file, dataset_cfg)
    elif scoring == "pope":
        output = score_pope(answers_file, dataset_cfg, question_file)
    elif scoring == "scienceqa":
        output = score_scienceqa(answers_file, dataset_cfg)
    elif scoring == "mmbench":
        output = score_mmbench(answers_file, dataset_cfg)
    elif scoring == "vqav2":
        output = score_vqav2(answers_file, dataset_cfg)

    print(output)

    # ── Save results ──
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"{exp_name}_{timestamp}.txt"

        meta = {
            "dataset":      args.dataset,
            "split":        args.split,
            "split_ratio":  args.split_ratio,
            "mode":         mode_tag,
            "n4": n4, "n2": n2, "n1": n1, "N": N,
            "use_quant":    use_quant,
            "budget":       args.budget,
            "timestamp":    timestamp,
        }

        with open(result_file, "w") as f:
            f.write(f"=== Config ===\n{json.dumps(meta, indent=2)}\n\n")
            f.write(f"=== Score ===\n{output}\n")

        print(f"\n[saved] {result_file}")


if __name__ == "__main__":
    main()
