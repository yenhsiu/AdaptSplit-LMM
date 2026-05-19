#!/usr/bin/env python3
"""
measure_latency_prumerge.py — 手法②④ 用
Sweeps N = 1..576 to measure Projector + LLM latency on a single device.

For each N, injects random visual tokens [1, N, 1024] directly into the
MLP Projector then concatenates text (40 tokens) and runs LlamaModel.
ViT and PruMerge are NOT included (their timings come from measure_edge.py).

This gives T_proj_llm(N) — the "after-PruMerge" portion of the pipeline
for every token count, enabling full T_total reconstruction in result_analysis.

Usage:
    python measure_latency_prumerge.py [--device "Jetson AGX Orin"]
    python measure_latency_prumerge.py --n-step 8    # sample every 8th N (72 points, faster)
    python measure_latency_prumerge.py --n-step 1    # all 576 values
Output:
    results_latency_pm.csv  (columns: component, N, mean_ms, std_ms)
"""

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

TEXT_TOKENS = 40
VIT_DIM     = 1024
PROJ_DIM    = 4096
N_MIN       = 1
N_MAX       = 576
DEFAULT_STEP = 8     # change to 1 for all 576 values (very slow on edge)
WARMUP      = 5
REPEAT      = 10

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, n_warmup, n_repeat):
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

def detect_system_info():
    info = {}
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    info["cuda_version"] = (torch.version.cuda or "N/A") if torch.cuda.is_available() else "N/A"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = "unknown"
    info["python"] = platform.python_version()
    info["torch"] = torch.__version__
    return info

# ── Model building ────────────────────────────────────────────────────────────

def build_projector(device):
    """LLaVA-1.5 MLP projector: Linear(1024,4096) + GELU + Linear(4096,4096)."""
    proj = nn.Sequential(
        nn.Linear(VIT_DIM, PROJ_DIM),
        nn.GELU(),
        nn.Linear(PROJ_DIM, PROJ_DIM),
    ).to(device=device, dtype=torch.float16).eval()
    print("  Built MLP Projector (1024→4096, random weights).", flush=True)
    return proj

def build_llm(device):
    from transformers import LlamaConfig, LlamaModel
    config = LlamaConfig(
        hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, max_position_embeddings=4096,
        vocab_size=32000,
    )
    print("  Building LlamaModel (Vicuna-7B arch, random weights) ...", flush=True)
    return LlamaModel(config).to(device=device, dtype=torch.float16).eval()

# ── Sweep measurement ─────────────────────────────────────────────────────────

def sweep_n(projector, llm, device, n_values):
    """
    For each N: random [1, N, 1024] → Projector → concat text(40) → LLM
    Returns dict {N: {"mean_ms": ..., "std_ms": ...}}
    """
    text_embeds = torch.randn(1, TEXT_TOKENS, PROJ_DIM, device=device, dtype=torch.float16)
    results = {}
    total = len(n_values)

    for idx, n in enumerate(n_values):
        vis = torch.randn(1, n, VIT_DIM, device=device, dtype=torch.float16)

        def run(v=vis):
            with torch.no_grad():
                proj_out = projector(v)
                inputs   = torch.cat([proj_out, text_embeds], dim=1)
                llm(inputs_embeds=inputs, use_cache=False)

        mean, std = timed_run(run, WARMUP, REPEAT)
        results[n] = {"mean_ms": mean, "std_ms": std}

        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"  [{idx+1:>4}/{total}] N={n:<4}  {mean:.1f} ± {std:.1f} ms", flush=True)

    return results

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_results(device_name, sys_info, n_values, results):
    hdr("裝置資訊")
    print(f"  裝置    : {device_name}")
    print(f"  GPU     : {sys_info['gpu']}")
    print(f"  CUDA    : {sys_info['cuda_version']}")
    print(f"  RAM     : {sys_info['ram_gb']} GB")
    print(f"  PyTorch : {sys_info['torch']}")

    hdr(f"T_proj_llm  (Projector + LLM, text={TEXT_TOKENS}, N={n_values[0]}..{n_values[-1]})")
    print(f"  {'N':<6} {'mean_ms':>10} {'std_ms':>8}")
    print(f"  {'─'*28}")
    for n in n_values:
        d = results[n]
        print(f"  {n:<6} {d['mean_ms']:>10.3f} {d['std_ms']:>8.3f} ms")
    print()

def export_csv(device_name, n_values, results):
    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / f"results_latency_pm_{safe}.csv")
    fields = ["component", "N", "mean_ms", "std_ms"]
    rows = [
        {"component": "T_proj_llm", "N": n,
         "mean_ms": results[n]["mean_ms"], "std_ms": results[n]["std_ms"]}
        for n in n_values
    ]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {fname}  ({len(rows)} rows)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Projector + LLM latency sweep over N visual tokens")
    parser.add_argument("--device", type=str, default="Unknown Device",
                        help='Label, e.g. "Jetson AGX Orin"')
    parser.add_argument("--n-step", type=int, default=DEFAULT_STEP,
                        help=f"Token count step size (default {DEFAULT_STEP}). "
                             f"Use 1 for all {N_MAX} values (very slow on edge).")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="Explicit list of N values to test (overrides --n-step)")
    args = parser.parse_args()

    if args.n_values:
        n_values = sorted(set(max(1, n) for n in args.n_values if 1 <= n <= N_MAX))
    else:
        n_values = list(range(N_MIN, N_MAX + 1, args.n_step))
        if N_MAX not in n_values:
            n_values.append(N_MAX)

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  measure_latency_prumerge.py  →  results_latency_pm.csv")
    print(SEP)
    print(f"  Device  : {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")
    print(f"  N range : {n_values[0]}..{n_values[-1]}  ({len(n_values)} values, step={args.n_step})")
    print(f"  Per N   : warmup={WARMUP}, repeat={REPEAT}")

    print("\n[1/3] Building Projector ...")
    projector = build_projector(device)

    print("\n[2/3] Building LLM ...")
    llm = build_llm(device)

    print(f"\n[3/3] Sweeping N = {n_values[0]}..{n_values[-1]} ...")
    results = sweep_n(projector, llm, device, n_values)

    print_results(args.device, sys_info, n_values, results)

    hdr("CSV Output")
    export_csv(args.device, n_values, results)

if __name__ == "__main__":
    main()
