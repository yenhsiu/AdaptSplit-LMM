#!/usr/bin/env python3
"""
Profile PruMerge output token count distribution across datasets.

Directly uses CLIPVisionTower from clip_encoder.py to ensure consistency with actual inference.
Model is loaded in float16 on GPU to match inference conditions.

Supported datasets:
  pope      – 500 COCO natural scene images
  mme       – 1187 images across 14 categories
  textvqa   – 5000 text-heavy images

Usage:
    python profile_token_count.py --dataset pope --method both --max-images 500
    python profile_token_count.py --dataset pope --method plus --debug --max-images 5
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

# ── Constants ─────────────────────────────────────────────────────────────────

CLIP_MODEL_DEFAULT = "openai/clip-vit-large-patch14-336"
FULL_TOKENS = 576  # 24×24 patches for 336-px input (no CLS)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "pope": {
        "jsonl":     Path("/home/yenhsiu/AdaptSplit-LMM/playground/data/eval/pope/llava_pope_test.jsonl"),
        "image_dir": Path("/mnt/ssd/yenhsiu_datasets/POPE/coco_val2014"),
        "img_field": "image",
    },
    "mme": {
        "jsonl":     Path("/home/yenhsiu/AdaptSplit-LMM/playground/data/eval/MME/llava_mme.jsonl"),
        "image_dir": Path("/home/yenhsiu/AdaptSplit-LMM/playground/data/eval/MME/MME_Benchmark_release_version"),
        "img_field": "image",
    },
    "textvqa": {
        "jsonl":     Path("/mnt/ssd/yenhsiu_datasets/textvqa/llava_textvqa_val_v051_ocr.jsonl"),
        "image_dir": Path("/mnt/ssd/yenhsiu_datasets/textvqa/train_images"),
        "img_field": "image",
    },
}

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_unique_images(dataset: str, max_images: int) -> list[Path]:
    cfg = DATASET_CONFIGS[dataset]
    seen, paths = set(), []
    with open(cfg["jsonl"]) as f:
        for line in f:
            fname = json.loads(line)[cfg["img_field"]]
            if fname not in seen:
                seen.add(fname)
                p = cfg["image_dir"] / fname
                if p.exists():
                    paths.append(p)
            if len(paths) >= max_images:
                break
    return paths

# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(vision_tower, pixel_values, method: str, debug: bool = False) -> dict:
    with torch.no_grad():
        if method == "plus":
            output = vision_tower.token_prune_merge_advanced_plus(
                pixel_values, if_adaptive=True, reduction_ratio=1/8
            )
        elif method == "advanced":
            output = vision_tower.token_prune_merge_advanced(
                pixel_values, if_adaptive=True, reduction_ratio=1/8
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    n_out = output.size(1)
    if debug:
        print(f"    [DEBUG] {method} output shape: {output.shape}, n_out={n_out}")
        print(f"    [DEBUG] vision_tower model: {vision_tower.vision_tower_name}")
    return {"n_out": n_out, "method": method}

# ── Stats ─────────────────────────────────────────────────────────────────────

def pct(data, p):
    return float(np.percentile(data, p))

def print_stats(label: str, values: list):
    if not values:
        return
    if len(values) == 1:
        n = values[0]
        print(f"\n  {label}: {n:.0f}  (compression: {(1 - n/FULL_TOKENS)*100:.1f}%,  {FULL_TOKENS} → {n:.0f} tokens)")
        return
    mean = statistics.mean(values)
    compression = (1 - mean / FULL_TOKENS) * 100
    saved = FULL_TOKENS - mean
    print(f"\n  {label}")
    print(f"    mean ± std  = {mean:.1f} ± {statistics.stdev(values):.1f}"
          f"  (compression {compression:.1f}%,  saves {saved:.1f} tokens vs {FULL_TOKENS})")
    print(f"    min / max   = {min(values):.0f} / {max(values):.0f}")
    print(f"    p5 / p25 / p50 / p75 / p95 = "
          f"{pct(values,5):.0f} / {pct(values,25):.0f} / {pct(values,50):.0f} / "
          f"{pct(values,75):.0f} / {pct(values,95):.0f}")

def print_buckets(n_outs: list, label: str):
    if not n_outs:
        return
    total = len(n_outs)
    print(f"\n  {label} — N_out bucket distribution (baseline={FULL_TOKENS}):")
    buckets = [(0,50),(50,100),(100,150),(150,200),(200,300),(300,400),(400,577)]
    for lo, hi in buckets:
        cnt = sum(lo <= n < hi for n in n_outs)
        bar = "█" * int(cnt / total * 40) if total > 0 else ""
        pct_val = cnt / total * 100
        print(f"    [{lo:>3}–{hi:<3})  {cnt:>4}  {pct_val:5.1f}%  {bar}")

def print_ttx(n_outs: list, label: str):
    """Transmission time estimate: N × 1024 dims × bits / bandwidth."""
    if not n_outs:
        return
    median_n = pct(n_outs, 50)
    p95_n    = pct(n_outs, 95)

    def t_ms(n, bits, bw_mbps):
        return n * 1024 * bits / (bw_mbps * 1e6) * 1000

    print(f"\n  T_tx estimate  [{label}]  (N × 1024 dims × bits / bandwidth)")
    print(f"  {'bits':>4}  {'BW':>6}  {'baseline':>10}  {'median':>10}  {'p95':>10}  {'saving@median':>14}")
    for bits in [4, 2, 1]:
        for bw in [1, 5, 20]:
            t_base   = t_ms(FULL_TOKENS, bits, bw)
            t_med    = t_ms(median_n,    bits, bw)
            t_p95    = t_ms(p95_n,       bits, bw)
            saving   = (1 - t_med / t_base) * 100
            print(f"  {bits:>4}bit  {bw:>3}Mbps  {t_base:>9.2f}ms  {t_med:>9.2f}ms"
                  f"  {t_p95:>9.2f}ms  {saving:>12.1f}%")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_method(vision_tower, processor, image_paths, method, device, debug=False):
    results = []
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            pv  = processor(images=img, return_tensors="pt").pixel_values.to(device, dtype=torch.float16)
            r   = count_tokens(vision_tower, pv, method=method, debug=(debug and i == 0))
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(image_paths)}] N_out={r['n_out']:4d}")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
    return results

def save_csv(results, method, out_path):
    if not results:
        print(f"  No results to save for {method}")
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "image_idx", "n_out"])
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow({"method": method, "image_idx": i, "n_out": r["n_out"]})
    print(f"  CSV saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["pope", "mme", "textvqa"], default="pope")
    parser.add_argument("--method",     choices=["advanced", "plus", "both"], default="both")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--no-plot",    action="store_true")
    parser.add_argument("--debug",      action="store_true", help="Print debug info for first image")
    parser.add_argument("--clip-model", type=str, default=CLIP_MODEL_DEFAULT,
                        help="CLIP model name or path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Method: {args.method}\n")

    vision_args = SimpleNamespace(
        mm_vision_select_layer=-2,
        mm_vision_select_feature='patch'
    )

    print(f"Loading {args.clip_model} …")
    vision_tower = CLIPVisionTower(args.clip_model, vision_args, delay_load=False)
    vision_tower.vision_tower = vision_tower.vision_tower.to(device=device, dtype=torch.float16)
    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    print("  Model loaded.\n")

    image_paths = load_unique_images(args.dataset, args.max_images)
    print(f"Images to process: {len(image_paths)}\n")

    SEP = "═" * 60
    methods = ["advanced", "plus"] if args.method == "both" else [args.method]
    all_results = {}

    for method in methods:
        print(f"{SEP}")
        print(f"  Method: token_prune_merge_{method}")
        print(SEP)
        results = run_method(vision_tower, processor, image_paths, method, device, debug=args.debug)
        all_results[method] = results

        n_outs = [r["n_out"] for r in results]
        print_stats(f"N_out  [{method}]", n_outs)
        print_buckets(n_outs, method)
        print_ttx(n_outs, method)

        out_csv = Path(__file__).parent / f"{args.dataset}_token_count_{method}.csv"
        save_csv(results, method, out_csv)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    if args.method == "both" and len(image_paths) > 1:
        adv  = [r["n_out"] for r in all_results["advanced"]]
        plus = [r["n_out"] for r in all_results["plus"]]
        print(f"\n{SEP}")
        print(f"  Comparison: advanced vs plus  (baseline={FULL_TOKENS})")
        print(SEP)
        print(f"  {'':20}  {'baseline':>10}  {'advanced':>10}  {'plus':>10}")
        for label, fn in [
            ("mean",   statistics.mean),
            ("median", lambda x: pct(x, 50)),
            ("p95",    lambda x: pct(x, 95)),
            ("max",    max),
        ]:
            print(f"  {label:<20}  {FULL_TOKENS:>10.1f}  {fn(adv):>10.1f}  {fn(plus):>10.1f}")
        print(f"  {'compression (mean)':<20}  {'—':>10}  "
              f"{(1-statistics.mean(adv)/FULL_TOKENS)*100:>9.1f}%  "
              f"{(1-statistics.mean(plus)/FULL_TOKENS)*100:>9.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot and len(image_paths) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_methods = len(all_results)
            fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 4))
            if n_methods == 1:
                axes = [axes]

            colors = {"advanced": "steelblue", "plus": "darkorange"}
            for i, (method, results) in enumerate(all_results.items()):
                n_outs = [r["n_out"] for r in results]
                ax = axes[i]
                ax.hist(n_outs, bins=30, color=colors[method], edgecolor="white", alpha=0.85)
                ax.axvline(statistics.mean(n_outs), color="red",   linestyle="--",
                           label=f"mean={statistics.mean(n_outs):.0f}")
                ax.axvline(pct(n_outs, 95),         color="black", linestyle=":",
                           label=f"p95={pct(n_outs,95):.0f}")
                ax.axvline(FULL_TOKENS, color="gray", linestyle="-", alpha=0.5,
                           label=f"baseline={FULL_TOKENS}")
                ax.set_xlabel("N_out (tokens after PruMerge)")
                ax.set_ylabel("Image count")
                ax.set_title(f"token_prune_merge_{method}  [{args.dataset}]")
                ax.legend()

            plt.tight_layout()
            out_png = Path(__file__).parent / f"{args.dataset}_token_count_{args.method}.png"
            plt.savefig(out_png, dpi=150)
            print(f"\n  Plot saved: {out_png}")
        except ImportError:
            print("  matplotlib not available, skipping plot")

if __name__ == "__main__":
    main()
