#!/usr/bin/env python3
"""
Cloud-side latency measurement.
Measures: T_cloud (LLM prefill latency for various N visual tokens)

Run on: GPU server
Does NOT load any pretrained weights — uses randomly initialised LlamaModel.
Latency depends only on architecture (num_layers, hidden_size, etc.), not weights.

Usage:
    python measure_cloud.py --device "RTX 3090"
    python measure_cloud.py --device "A100"
"""

import argparse
import csv
import platform
import statistics
import time
from typing import Optional

import torch
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────

N_VALUES     = [15, 19, 25, 35, 50, 75, 96, 150]
PROJ_DIM     = 4096      # LLaVA projector output dim (1024 → 4096)
WARMUP       = 10
REPEAT_CLOUD = 50

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_run(fn, n_warmup: int, n_repeat: int) -> tuple[float, float]:
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
    info["torch"]  = torch.__version__
    return info

# ── Model loading ─────────────────────────────────────────────────────────────

def build_llama_model(device: torch.device):
    """
    LlamaModel (Vicuna-7B architecture) with random weights.
    Identical architecture to lmsys/vicuna-7b-v1.5:
      32 layers, hidden=4096, heads=32, intermediate=11008.
    No checkpoint download needed.
    """
    from transformers import LlamaConfig, LlamaModel

    config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
        vocab_size=32000,
    )
    print("  Building LlamaModel (Vicuna-7B arch, random weights) …", flush=True)
    model = LlamaModel(config).to(device=device, dtype=torch.float16).eval()
    return model, "LlamaModel(Vicuna-7B arch, random weights)"

# ── T_cloud ───────────────────────────────────────────────────────────────────

def measure_t_cloud(model, device: torch.device) -> dict[int, dict]:
    """
    Measure LLM prefill latency for each N.
    Input: (1, N, 4096) inputs_embeds — simulates Projector output fed into LLM.
    """
    results: dict[int, dict] = {}

    for n in N_VALUES:
        embeds = torch.randn(1, n, PROJ_DIM, device=device, dtype=torch.float16)

        def run(e=embeds):
            with torch.no_grad():
                model(inputs_embeds=e, use_cache=False)

        mean, std = timed_run(run, WARMUP, REPEAT_CLOUD)
        results[n] = {"mean_ms": mean, "std_ms": std}

    return results

# ── Linearity check ───────────────────────────────────────────────────────────

def check_linearity(results: dict[int, dict]) -> dict:
    """
    Simple linear regression on T_cloud vs N.
    Returns slope (ms/token), intercept, R².
    """
    xs = [n for n in N_VALUES if results[n]["mean_ms"] is not None]
    ys = [results[n]["mean_ms"] for n in xs]

    n  = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    ss_xx = sum((x - mx) ** 2 for x in xs)
    slope     = ss_xy / ss_xx if ss_xx > 0 else 0.0
    intercept = my - slope * mx
    y_pred    = [slope * x + intercept for x in xs]
    ss_res    = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot    = sum((y - my) ** 2 for y in ys)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {"slope_ms_per_token": slope, "intercept_ms": intercept, "r2": r2}

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def row(label: str, mean: Optional[float], std: Optional[float], note: str = ""):
    if mean is None:
        print(f"  {label:<30} FAILED  {note}")
    else:
        print(f"  {label:<30} {mean:8.3f} ± {std:6.3f} ms  {note}")

def print_results(device_name, sys_info, model_label, results, lin):
    hdr("裝置資訊")
    print(f"  裝置      : {device_name}")
    print(f"  GPU       : {sys_info['gpu']}")
    print(f"  CUDA      : {sys_info['cuda_version']}")
    print(f"  RAM       : {sys_info['ram_gb']} GB")
    print(f"  PyTorch   : {sys_info['torch']}")

    hdr(f"T_cloud  （LLM prefill）")
    print(f"  模型：{model_label}")
    print()
    for n in N_VALUES:
        d = results[n]
        row(f"N={n}", d["mean_ms"], d["std_ms"])

    hdr("線性關係分析  T_cloud(N)")
    print(f"  slope     : {lin['slope_ms_per_token']:+.4f} ms / token")
    print(f"  intercept : {lin['intercept_ms']:.3f} ms")
    print(f"  R²        : {lin['r2']:.4f}", end="")
    if lin["r2"] > 0.99:
        print("  ← 線性關係成立")
    elif lin["r2"] > 0.95:
        print("  ← 大致線性")
    else:
        print("  ← 非線性，需關注")
    print()

def export_csv(device_name, model_label, results):
    fname = f"{device_name.replace(' ','_')}_cloud_latency.csv"
    rows  = []
    for n in N_VALUES:
        d = results[n]
        rows.append({
            "component": "T_cloud",
            "N": n, "bits": "", "bw_mbps": "",
            "mean_ms": d["mean_ms"], "std_ms": d["std_ms"],
            "note": model_label,
        })
    fields = ["component", "N", "bits", "bw_mbps", "mean_ms", "std_ms", "note"]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV 已儲存：{fname}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cloud Latency Measurement")
    parser.add_argument("--device", type=str, default="Unknown GPU")
    args = parser.parse_args()

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  Cloud Latency Measurement")
    print(SEP)
    print(f"  裝置: {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")

    print("\n[1/2] 建立 LlamaModel (Vicuna-7B 架構，隨機權重) …")
    model, model_label = build_llama_model(device)

    print("\n[2/2] 測量 T_cloud …")
    results = measure_t_cloud(model, device)

    lin = check_linearity(results)

    print_results(args.device, sys_info, model_label, results, lin)

    hdr("CSV 輸出")
    export_csv(args.device, model_label, results)

if __name__ == "__main__":
    main()