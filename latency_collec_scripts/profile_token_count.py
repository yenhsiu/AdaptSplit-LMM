#!/usr/bin/env python3
"""
Profile PruMerge output token count distribution across datasets.

Supports two methods:
  advanced  – token_prune_merge_advanced   (TopK + 1 extra, no spatial)
  plus      – token_prune_merge_advanced_plus (TopK + spatial supplement + 1)

Supported datasets:
  pope      – 500 COCO natural scene images
  mme       – 1187 images across 14 categories (OCR, code, landmark …)
  textvqa   – 5000 text-heavy images (signs, documents …)

Usage:
    python profile_token_count.py --dataset pope --method both
    python profile_token_count.py --dataset mme  --method both
    python profile_token_count.py --dataset textvqa --method advanced
    python profile_token_count.py --dataset pope --method both --max-images 100
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────

CLIP_MODEL = "openai/clip-vit-large-patch14-336"

DATASET_CONFIGS = {
    "pope": {
        "jsonl":      Path("/home/yenhsiu/AdaptSplit-LMM/playground/data/eval/pope/llava_pope_test.jsonl"),
        "image_dir":  Path("/mnt/ssd/yenhsiu_datasets/POPE/coco_val2014"),
        "img_field":  "image",          # value is plain filename
        "subfolder":  False,
    },
    "mme": {
        "jsonl":      Path("/home/yenhsiu/AdaptSplit-LMM/playground/data/eval/MME/llava_mme.jsonl"),
        "image_dir":  Path("/home/yenhsiu/AdaptSplit-LMM/playground/data/eval/MME/MME_Benchmark_release_version"),
        "img_field":  "image",          # value is "category/filename.png"
        "subfolder":  True,
    },
    "textvqa": {
        "jsonl":      Path("/mnt/ssd/yenhsiu_datasets/textvqa/llava_textvqa_val_v051_ocr.jsonl"),
        "image_dir":  Path("/mnt/ssd/yenhsiu_datasets/textvqa/train_images"),
        "img_field":  "image",          # value is plain filename
        "subfolder":  False,
    },
}

# ── Hook helpers ──────────────────────────────────────────────────────────────

_hook_store: dict = {}

def _hook_k(module, input, output): _hook_store["k"] = output
def _hook_q(module, input, output): _hook_store["q"] = output

# ── outlier_detection (mirrors clip_encoder.py) ───────────────────────────────

def outlier_detection(attn: torch.Tensor) -> float:
    arr = attn.to(torch.float32).cpu().numpy().flatten()
    Q1 = float(np.percentile(arr, 25))
    Q3 = float(np.percentile(arr, 75))
    upper = Q3 + 1.5 * (Q3 - Q1)
    return float(np.sum(arr > upper)) / len(arr)

# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(
    model: CLIPVisionModel,
    pixel_values: torch.Tensor,
    method: str,
    select_layer: int = -2,
) -> dict:
    """
    Run ViT forward + PruMerge token selection, return counts only.

    method="advanced" → token_prune_merge_advanced   (n_out = n_topk + 1)
    method="plus"     → token_prune_merge_advanced_plus (n_out = n_topk + n_spatial + 1)
    """
    device = pixel_values.device

    last_layer = model.vision_model.encoder.layers[23]
    hk = last_layer.self_attn.k_proj.register_forward_hook(_hook_k)
    hq = last_layer.self_attn.q_proj.register_forward_hook(_hook_q)

    with torch.no_grad():
        out = model(pixel_values=pixel_values, output_hidden_states=True)

    hk.remove()
    hq.remove()

    # N = 576 patch tokens (drop CLS), C = 1024
    hidden = out.hidden_states[select_layer]   # (1, 577, 1024)
    _, N1, C = hidden.shape
    N = N1 - 1  # 576

    q, k = _hook_store["q"], _hook_store["k"]
    attn     = F.softmax((q @ k.transpose(-2, -1)) * (C ** -0.5), dim=-1)
    cls_attn = attn[:, 0, 1:]  # (1, 576)  — mirrors clip_encoder.py:201

    ratio  = outlier_detection(cls_attn)
    n_topk = int(N * ratio)

    if method == "advanced":
        # token_prune_merge_advanced: TopK + 1 merged extra token
        # n_spatial = 0 (no spatial supplement)
        return {"n_topk": n_topk, "n_spatial": 0, "n_out": n_topk + 1, "ratio": ratio}

    # method == "plus": token_prune_merge_advanced_plus
    _, idx   = torch.topk(cls_attn, n_topk, dim=1, largest=True)
    idx_set  = set(idx.flatten().tolist())

    # Spatial supplement — mirrors clip_encoder.py:209-213
    step_length = int(1 / ratio) if ratio > 0 else N
    spacing     = max(int(step_length / 3), 1)   # guard against spacing=0
    spatial_seq = torch.arange(0, 575, spacing, device=device)
    n_spatial   = sum(1 for x in spatial_seq.tolist() if x not in idx_set)

    return {"n_topk": n_topk, "n_spatial": n_spatial, "n_out": n_topk + n_spatial + 1, "ratio": ratio}

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_unique_images(dataset: str, max_images: int) -> list[Path]:
    cfg = DATASET_CONFIGS[dataset]
    seen, paths = set(), []
    with open(cfg["jsonl"]) as f:
        for line in f:
            fname = json.loads(line)[cfg["img_field"]]
            if fname not in seen:
                seen.add(fname)
                p = cfg["image_dir"] / fname   # subfolder paths work for both cases
                if p.exists():
                    paths.append(p)
            if len(paths) >= max_images:
                break
    return paths

# ── Stats ─────────────────────────────────────────────────────────────────────

def pct(data, p): return float(np.percentile(data, p))

def print_stats(label: str, values: list):
    print(f"\n  {label}")
    print(f"    mean ± std  = {statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}")
    print(f"    min / max   = {min(values):.0f} / {max(values):.0f}")
    print(f"    p5 / p25 / p50 / p75 / p95 = "
          f"{pct(values,5):.0f} / {pct(values,25):.0f} / {pct(values,50):.0f} / "
          f"{pct(values,75):.0f} / {pct(values,95):.0f}")

def print_buckets(n_outs: list, label: str):
    total = len(n_outs)
    print(f"\n  {label} — N_out bucket distribution:")
    buckets = [(0,50),(50,100),(100,150),(150,200),(200,300),(300,400),(400,577)]
    for lo, hi in buckets:
        cnt = sum(lo <= n < hi for n in n_outs)
        bar = "█" * int(cnt / total * 40)
        print(f"    [{lo:>3}–{hi:<3})  {cnt:>4}  {cnt/total*100:5.1f}%  {bar}")

def print_ttx(n_outs: list, label: str):
    median_n = pct(n_outs, 50)
    p95_n    = pct(n_outs, 95)
    print(f"\n  T_tx estimate  [{label}]  (N × 1024 × bits / bandwidth)")
    print(f"  {'bits':>4}  {'BW':>6}  {'median':>10}  {'p95':>10}")
    for bits in [4, 2, 1]:
        for bw in [1, 5, 20]:
            t_med = median_n * 1024 * bits / (bw * 1e6) * 1000
            t_p95 = p95_n   * 1024 * bits / (bw * 1e6) * 1000
            print(f"  {bits:>4}bit  {bw:>3}Mbps  {t_med:>9.2f}ms  {t_p95:>9.2f}ms")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_method(model, processor, image_paths, method, device):
    results = []
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        pv  = processor(images=img, return_tensors="pt").pixel_values.to(device)
        r   = count_tokens(model, pv, method=method)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(image_paths)}] N_out={r['n_out']:4d}  ratio={r['ratio']:.3f}")
    return results

def save_csv(results, method, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","image_idx","ratio","n_topk","n_spatial","n_out"])
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow({"method": method, "image_idx": i, **r})
    print(f"  CSV saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["pope","mme","textvqa"], default="pope")
    parser.add_argument("--method",     choices=["advanced","plus","both"], default="both")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--no-plot",    action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Method: {args.method}")

    print(f"Loading {CLIP_MODEL} …")
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    model     = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(device).eval()
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
        results = run_method(model, processor, image_paths, method, device)
        all_results[method] = results

        n_outs = [r["n_out"] for r in results]
        print_stats(f"N_out  [{method}]", n_outs)
        print_stats(f"ratio  [{method}]", [r["ratio"] for r in results])
        print_buckets(n_outs, method)
        print_ttx(n_outs, method)

        out_csv = Path(__file__).parent / f"{args.dataset}_token_count_{method}.csv"
        save_csv(results, method, out_csv)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    if args.method == "both":
        adv  = [r["n_out"] for r in all_results["advanced"]]
        plus = [r["n_out"] for r in all_results["plus"]]
        print(f"\n{SEP}")
        print(f"  Comparison: advanced vs plus")
        print(SEP)
        print(f"  {'':20}  {'advanced':>10}  {'plus':>10}")
        for label, fn in [("mean", statistics.mean), ("median", lambda x: pct(x,50)),
                          ("p95",  lambda x: pct(x,95)), ("max",  max)]:
            print(f"  {label:<20}  {fn(adv):>10.1f}  {fn(plus):>10.1f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_methods = len(all_results)
            fig, axes = plt.subplots(1, n_methods + 1, figsize=(6 * (n_methods + 1), 4))
            if n_methods == 1:
                axes = [axes, axes]  # make indexable

            colors = {"advanced": "steelblue", "plus": "darkorange"}
            for i, (method, results) in enumerate(all_results.items()):
                n_outs = [r["n_out"] for r in results]
                ax = axes[i]
                ax.hist(n_outs, bins=30, color=colors[method], edgecolor="white", alpha=0.85)
                ax.axvline(statistics.mean(n_outs), color="red",    linestyle="--",
                           label=f"mean={statistics.mean(n_outs):.0f}")
                ax.axvline(pct(n_outs, 95),         color="black",  linestyle=":",
                           label=f"p95={pct(n_outs,95):.0f}")
                ax.set_xlabel("N_out (tokens after PruMerge)")
                ax.set_ylabel("Image count")
                ax.set_title(f"token_prune_merge_{method}  [{args.dataset}]")
                ax.legend()

            # Ratio distribution (shared, same for both methods)
            last_method = list(all_results.keys())[-1]
            ratios = [r["ratio"] for r in all_results[last_method]]
            n_img  = len(list(all_results.values())[0])
            axes[-1].hist(ratios, bins=30, color="gray", edgecolor="white", alpha=0.85)
            axes[-1].set_xlabel("Adaptive reduction ratio")
            axes[-1].set_ylabel("Image count")
            axes[-1].set_title(f"outlier_detection ratio  [{args.dataset}, n={n_img}]")

            plt.tight_layout()
            out_png = Path(__file__).parent / f"{args.dataset}_token_count_{args.method}.png"
            plt.savefig(out_png, dpi=150)
            print(f"\n  Plot saved: {out_png}")
        except ImportError:
            print("  matplotlib not available, skipping plot")

if __name__ == "__main__":
    main()
