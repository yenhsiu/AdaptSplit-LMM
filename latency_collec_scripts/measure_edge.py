#!/usr/bin/env python3
"""
Edge-side latency measurement.
Measures: T_edge (ViT only), T_vit_prumerge (ViT + PruMerge_advanced + PruMerge_plus), T_quant, T_tx

Model architecture matches actual inference (openai/clip-vit-large-patch14-336, float16).
Does NOT load pretrained weights or dataset — random float16 input is sufficient for compute-time profiling.

Usage:
    python measure_edge.py --device "Jetson AGX Orin"
    python measure_edge.py --device "Raspberry Pi 4B" --skip-prumerge
"""

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn

# ── Repo path ─────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

# N_VALUES covers actual PruMerge output range:
#   token_prune_merge_advanced      → ~19 tokens  (float16, POPE)
#   token_prune_merge_advanced_plus → ~96 tokens  (float16, POPE)
N_VALUES      = [15, 19, 25, 35, 50, 75, 96, 150]
BIT_VALUES    = [4, 2, 1]
BW_VALUES     = [1, 5, 20]       # Mbps
TOKEN_DIM     = 1024
WARMUP        = 10
REPEAT_EDGE   = 100
REPEAT_PM     = 20
REPEAT_QUANT  = 100
PRUMERGE_BITS = 1

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, n_warmup: int, n_repeat: int) -> Tuple[float, float]:
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_repeat):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times), statistics.stdev(times) if len(times) > 1 else 0.0

def detect_system_info() -> dict:
    info: dict = {}
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    info["cuda_version"] = torch.version.cuda or "N/A" if torch.cuda.is_available() else "N/A"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = "unknown"
    info["python"] = platform.python_version()
    info["torch"]  = torch.__version__
    return info

# ── T_edge ────────────────────────────────────────────────────────────────────

def measure_t_edge(device: torch.device) -> dict:
    """
    CLIPVisionModel (ViT-L/14-336) with random weights, float16.
    Architecture matches openai/clip-vit-large-patch14-336.
    No pretrained download needed.
    """
    from transformers import CLIPVisionModel, CLIPVisionConfig

    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        image_size=336, patch_size=14, projection_dim=768,
    )
    print("  Building CLIPVisionModel (ViT-L/14-336, random weights, float16) …", flush=True)
    model = CLIPVisionModel(config).to(device=device, dtype=torch.float16).eval()
    pv = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)

    def run():
        with torch.no_grad():
            model(pixel_values=pv, output_hidden_states=True)

    mean, std = timed_run(run, WARMUP, REPEAT_EDGE)
    return {"mean_ms": mean, "std_ms": std,
            "model": "CLIPVisionModel(ViT-L/14-336, random)", "fallback": False}

# ── T_vit_prumerge ────────────────────────────────────────────────────────────

def measure_t_vit_prumerge(device: torch.device) -> dict:
    """
    CLIPVisionTower.forward() — same function as actual inference.
    Calls token_prune_merge_advanced then token_prune_merge_advanced_plus (as in forward()).
    Random weights, float16. n_out read directly from output tensor.
    """
    try:
        from llava.model.multimodal_encoder.clip_encoder_t import CLIPVisionTower
        from turboquant.compressors_v3 import MSECompressor
    except ImportError as e:
        print(f"  [WARN] Import failed: {e}")
        return {"mean_ms": None, "std_ms": None, "n_out": None}

    try:
        print("  Building CLIPVisionTower (ViT-L + PruMerge, random weights, float16) …", flush=True)
        tower = CLIPVisionTower.__new__(CLIPVisionTower)
        nn.Module.__init__(tower)
        tower.is_loaded         = False
        tower.vision_tower_name = "openai/clip-vit-large-patch14-336"
        tower.select_layer      = -2
        tower.select_feature    = "patch"
        tower.total_tokens      = 0

        from transformers import CLIPVisionModel, CLIPVisionConfig
        config = CLIPVisionConfig(
            hidden_size=1024, intermediate_size=4096,
            num_hidden_layers=24, num_attention_heads=16,
            image_size=336, patch_size=14, projection_dim=768,
        )
        tower.image_processor = None
        tower.vision_tower    = CLIPVisionModel(config)
        tower.vision_tower.requires_grad_(False)

        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        tower.compressor = MSECompressor(head_dim=1024, bits=PRUMERGE_BITS,
                                         seed=42, device=dev_str)
        tower.is_loaded = True
        tower = tower.to(device=device, dtype=torch.float16).eval()
    except Exception as e:
        print(f"  [WARN] CLIPVisionTower setup failed: {e}")
        return {"mean_ms": None, "std_ms": None, "n_out": None}

    pv = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)

    # Warm-up
    tower.latency_log = []
    for _ in range(3):
        with torch.no_grad():
            tower(pv)

    n_out = tower.latency_log[-1]["left_tokens"] + 1
    ratio = tower.latency_log[-1]["reduction_ratio"]
    print(f"  PruMerge output: {n_out} tokens (ratio={ratio:.3f})", flush=True)

    # Timed runs
    tower.latency_log = []
    total_times = []
    for _ in range(REPEAT_PM):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            tower(pv)
        sync()
        total_times.append((time.perf_counter() - t0) * 1000.0)

    # Per-stage breakdown from latency_log
    stage_keys = ["t_vit_ms", "t_attn_topk_ms", "t_merge_loop_ms"]
    breakdown = {}
    for k in stage_keys:
        vals = [e[k] for e in tower.latency_log]
        breakdown[k] = {"mean_ms": sum(vals)/len(vals),
                        "std_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    mean_total = sum(total_times) / len(total_times)
    std_total  = statistics.stdev(total_times) if len(total_times) > 1 else 0.0

    breakdown["t_quant_derived_ms"] = max(
        mean_total
        - breakdown["t_vit_ms"]["mean_ms"]
        - breakdown["t_attn_topk_ms"]["mean_ms"]
        - breakdown["t_merge_loop_ms"]["mean_ms"],
        0.0
    )

    return {
        "mean_ms": mean_total, "std_ms": std_total,
        "n_out": n_out, "reduction_ratio": ratio,
        "breakdown": breakdown,
    }

# ── T_quant ───────────────────────────────────────────────────────────────────

def measure_t_quant(device: torch.device) -> Dict[Tuple[int, int], Dict]:
    """MSECompressor.compress() + decompress() for all (N, bits) combos."""
    try:
        from turboquant.compressors_v3 import MSECompressor
    except ImportError as e:
        print(f"  [WARN] MSECompressor import failed: {e}")
        return {}

    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    results: Dict[Tuple[int, int], Dict] = {}

    for bits in BIT_VALUES:
        c = MSECompressor(head_dim=TOKEN_DIM, bits=bits, seed=42, device=dev_str)
        for n in N_VALUES:
            tok = torch.randn(1, 1, n, TOKEN_DIM, device=device)
            def run(t=tok, comp=c):
                comp.decompress(comp.compress(t))
            mean, std = timed_run(run, WARMUP, REPEAT_QUANT)
            results[(n, bits)] = {"mean_ms": mean, "std_ms": std}

    return results

# ── T_tx ─────────────────────────────────────────────────────────────────────

def calc_t_tx(n: int, bits: int, bw_mbps: float) -> float:
    return n * TOKEN_DIM * bits / (bw_mbps * 1_000_000) * 1000.0

def build_t_tx_table() -> List[Dict]:
    return [{"N": n, "bits": b, "bw_mbps": s, "t_tx_ms": calc_t_tx(n, b, s)}
            for n in N_VALUES for b in BIT_VALUES for s in BW_VALUES]

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def row(label: str, mean: Optional[float], std: Optional[float], note: str = ""):
    if mean is None:
        print(f"  {label:<34} FAILED  {note}")
    else:
        print(f"  {label:<34} {mean:8.3f} ± {std:6.3f} ms  {note}")

def print_results(device_name, sys_info, t_edge, t_pm, quant, tx_table):
    hdr("裝置資訊")
    print(f"  裝置      : {device_name}")
    print(f"  GPU       : {sys_info['gpu']}")
    print(f"  CUDA      : {sys_info['cuda_version']}")
    print(f"  RAM       : {sys_info['ram_gb']} GB")
    print(f"  PyTorch   : {sys_info['torch']}")

    hdr("T_edge  （CLIPVisionModel，純 ViT，random weights，float16）")
    fb = "[fallback: ViT-B]" if t_edge.get("fallback") else ""
    row("T_edge", t_edge["mean_ms"], t_edge["std_ms"], fb)

    hdr("T_vit_prumerge  （CLIPVisionTower.forward()，ViT + PruMerge_advanced + PruMerge_plus）")
    n_out = t_pm.get("n_out")
    note  = f"→ {n_out} tokens" if n_out else ""
    row("總計", t_pm["mean_ms"], t_pm["std_ms"], note)
    bd = t_pm.get("breakdown", {})
    if bd:
        print()
        for label, key in [
            ("  ├ Stage 1  ViT forward",       "t_vit_ms"),
            ("  ├ Stage 2  Attn+TopK+Gather",  "t_attn_topk_ms"),
            ("  ├ Stage 3  Merge loop",         "t_merge_loop_ms"),
        ]:
            d = bd[key]
            row(label, d["mean_ms"], d["std_ms"])
        tq = bd.get("t_quant_derived_ms", 0.0)
        print(f"  {'  └ Stage 4  TurboQuant（推算）':<34} {tq:8.3f} ms")

    hdr("T_quant  （MSECompressor，各 N × bits 組合）")
    if not quant:
        print("  [FAILED]")
    else:
        for bits in BIT_VALUES:
            print(f"\n  {bits}bit:")
            for n in N_VALUES:
                d = quant[(n, bits)]
                row(f"    N={n}", d["mean_ms"], d["std_ms"])

    hdr("T_tx  （傳輸時間，公式計算）")
    print(f"  {'N':>4}  {'bits':>4}  {'BW(Mbps)':>9}  {'T_tx(ms)':>10}")
    print(f"  {'─'*40}")
    for r in tx_table:
        print(f"  {r['N']:>4}  {r['bits']:>4}  {r['bw_mbps']:>9.1f}  {r['t_tx_ms']:>10.4f}")
    print()

def export_csv(device_name, t_edge, t_pm, quant, tx_table):
    fname = f"{device_name.replace(' ','_')}_edge_latency.csv"
    rows  = []

    def add(comp, N, bits, bw, mean, std, note=""):
        rows.append({"component": comp, "N": N, "bits": bits,
                     "bw_mbps": bw, "mean_ms": mean, "std_ms": std, "note": note})

    add("T_edge", "", "", "",
        t_edge["mean_ms"], t_edge["std_ms"],
        t_edge["model"] + (" [fallback]" if t_edge["fallback"] else ""))

    add("T_vit_prumerge", t_pm.get("n_out", ""), "", "",
        t_pm["mean_ms"], t_pm["std_ms"], "CLIPVisionTower.forward()")

    for (n, b), d in quant.items():
        add("T_quant", n, b, "", d["mean_ms"], d["std_ms"], "MSECompressor")

    for r in tx_table:
        add("T_tx", r["N"], r["bits"], r["bw_mbps"], r["t_tx_ms"], 0.0, "calculated")

    fields = ["component", "N", "bits", "bw_mbps", "mean_ms", "std_ms", "note"]
    with open(fname, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows)

    print(f"  CSV 已儲存：{fname}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Edge Latency Measurement")
    parser.add_argument("--device", type=str, default="Unknown Device")
    parser.add_argument("--skip-edge",     action="store_true")
    parser.add_argument("--skip-prumerge", action="store_true",
                        help="跳過 T_vit_prumerge（記憶體不足時使用）")
    args = parser.parse_args()

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  Edge Latency Measurement")
    print(SEP)
    print(f"  裝置: {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")

    if args.skip_edge:
        t_edge = {"mean_ms": None, "std_ms": None, "model": "skipped", "fallback": False}
    else:
        print("\n[1/4] T_edge …")
        t_edge = measure_t_edge(device)

    if args.skip_prumerge:
        t_pm = {"mean_ms": None, "std_ms": None, "n_out": None}
    else:
        print("\n[2/4] T_vit_prumerge …")
        t_pm = measure_t_vit_prumerge(device)

    print("\n[3/4] T_quant …")
    quant = measure_t_quant(device)

    print("\n[4/4] T_tx（公式計算）")
    tx_table = build_t_tx_table()

    print_results(args.device, sys_info, t_edge, t_pm, quant, tx_table)
    hdr("CSV 輸出")
    export_csv(args.device, t_edge, t_pm, quant, tx_table)

if __name__ == "__main__":
    main()
