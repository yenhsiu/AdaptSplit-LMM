#!/usr/bin/env python3
"""
LMM Split Inference Latency Measurement Script
Uses real CLIPVisionTower (PruMerge) and MSECompressor (TurboQuant).

Measurement breakdown:
  T_edge          : CLIPVisionModel only (no PruMerge), 100 runs
  T_vit_prumerge  : CLIPVisionTower.forward() = ViT + PruMerge + TurboQuant, 20 runs
  T_quant         : MSECompressor.compress() + decompress() per (N, bits), 100 runs
  T_prumerge      : T_vit_prumerge - T_edge - T_quant(N_actual, bits=1)  [derived]
  T_tx            : formula, no actual transmission
  T_cloud         : LLaVA-PruMerge LoRA (real) or LlamaForCausalLM (random weights, same arch), 50 runs
"""

import argparse
import csv
import platform
import sys
import os
import time
import statistics
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn

# ── Repo paths ────────────────────────────────────────────────────────────────

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Config ────────────────────────────────────────────────────────────────────

N_VALUES        = [15, 19, 25, 35, 50, 75, 96, 150]
BIT_VALUES      = [4, 2, 1]
BW_VALUES       = [1, 5, 20]            # Mbps
TOKEN_DIM       = 1024                   # CLIP hidden dim
PROJ_DIM        = 4096                   # LLaVA projector output dim
WARMUP          = 10
REPEAT_EDGE     = 100
REPEAT_VIT_PM   = 20                     # CLIPVisionTower is slow, fewer reps
REPEAT_QUANT    = 100
REPEAT_CLOUD    = 50

# PruMerge adaptive mode typically outputs ~N=35 tokens (reduction_ratio ≈ 1/16)
# We time T_vit_prumerge once and derive T_prumerge via subtraction.
PRUMERGE_BITS   = 1                      # bits used in CLIPVisionTower.load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_run(fn, n_warmup: int, n_repeat: int) -> Tuple[float, float]:
    """Warm-up then time fn; return (mean_ms, std_ms)."""
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

    return (
        sum(times) / len(times),
        statistics.stdev(times) if len(times) > 1 else 0.0,
    )


def detect_system_info() -> dict:
    info: dict = {}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "unknown"
    else:
        info["gpu"] = "CPU only"
        info["cuda_version"] = "N/A"
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
    No pretrained download needed — latency depends only on architecture.
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
        print(f"  [WARN] Cannot import CLIPVisionTower: {e}")
        return {"mean_ms": None, "std_ms": None, "n_out": None, "note": "IMPORT_FAILED",
                "breakdown": {}}

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
        return {"mean_ms": None, "std_ms": None, "n_out": None, "note": str(e),
                "breakdown": {}}

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
    for _ in range(REPEAT_VIT_PM):
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
    """
    Real MSECompressor.compress() + decompress() for each (N, bits).
    Input tensor shape: (1, 1, N, 1024)  — same as CLIPVisionTower uses.
    """
    try:
        from turboquant.compressors_v3 import MSECompressor
    except ImportError as e:
        print(f"  [WARN] Cannot import MSECompressor: {e}")
        return {}

    results: Dict[Tuple[int, int], Dict] = {}
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    for bits in BIT_VALUES:
        compressor = MSECompressor(head_dim=TOKEN_DIM, bits=bits, seed=42, device=dev_str)

        for n in N_VALUES:
            tokens = torch.randn(1, 1, n, TOKEN_DIM, device=device)

            def run(tok=tokens, c=compressor):
                compressed = c.compress(tok)
                c.decompress(compressed)

            mean, std = timed_run(run, WARMUP, REPEAT_QUANT)
            results[(n, bits)] = {"mean_ms": mean, "std_ms": std}

    return results


# ── T_tx ──────────────────────────────────────────────────────────────────────

def calc_t_tx(n: int, bits: int, bandwidth_mbps: float) -> float:
    """T_tx(ms) = N × 1024 × B / (S × 1_000_000) × 1000"""
    return n * TOKEN_DIM * bits / (bandwidth_mbps * 1_000_000) * 1000.0


def build_t_tx_table() -> List[Dict]:
    rows = []
    for n in N_VALUES:
        for b in BIT_VALUES:
            for s in BW_VALUES:
                rows.append({"N": n, "bits": b, "bw_mbps": s,
                              "t_tx_ms": calc_t_tx(n, b, s)})
    return rows


# ── T_cloud ───────────────────────────────────────────────────────────────────

# LLaVA-PruMerge LoRA checkpoint (matches pope.sh / mme.sh / textvqa.sh)
LORA_CKPT   = "/mnt/ssd/yuzhang_models/llava-prumerge-vicuna-7b-v1.5-lora"
VICUNA_BASE = "lmsys/vicuna-7b-v1.5"


def _build_random_llm(device: torch.device) -> nn.Module:
    """LlamaForCausalLM with Vicuna-7B architecture and randomly initialized weights."""
    from transformers import LlamaForCausalLM, LlamaConfig
    config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        vocab_size=32000,
        max_position_embeddings=4096,
    )
    print("  Building LlamaForCausalLM (Vicuna-7B arch, random weights, float16) …", flush=True)
    return LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()


def measure_t_cloud(device: torch.device) -> Dict[int, Dict]:
    """
    Tier 1: load LlavaLlamaForCausalLM from local LoRA checkpoint + Vicuna-7B base.
    Tier 2: if weights not found, build LlamaForCausalLM with random weights (same arch).
    Returns empty dict if both fail.

    Prefill is measured by passing (1, N, 4096) inputs_embeds directly to the LM.
    """
    mode        = "random"   # "real" | "random"
    model_label = "LlamaForCausalLM (Vicuna-7B arch, random weights)"
    llm         = None

    # Tier 1 — real LoRA weights
    try:
        from llava.model.builder import load_pretrained_model

        print(f"  Loading Vicuna-7B base + LLaVA-PruMerge LoRA …", flush=True)
        print(f"    base : {VICUNA_BASE}", flush=True)
        print(f"    lora : {LORA_CKPT}", flush=True)

        _, llava, _, _ = load_pretrained_model(
            model_path=LORA_CKPT,
            model_base=VICUNA_BASE,
            model_name="llava-lora-prunemerge",
            device=str(device),
            device_map={"": device},
        )
        llm         = llava.eval()
        mode        = "real"
        model_label = "LLaVA-PruMerge LoRA (Vicuna-7B)"

    except Exception as e:
        print(f"  [WARN] LLaVA-PruMerge load failed: {e}", flush=True)

        # Tier 2 — same architecture, random weights
        try:
            llm = _build_random_llm(device)
        except Exception as e2:
            print(f"  [ERROR] Random LLM build also failed: {e2}", flush=True)
            return {}

    results: Dict[int, Dict] = {}
    for n in N_VALUES:
        tokens = torch.randn(1, n, PROJ_DIM, device=device, dtype=torch.float16)
        def run(tok=tokens):
            with torch.no_grad():
                llm.model(inputs_embeds=tok, use_cache=False)

        mean, std = timed_run(run, max(WARMUP, 5), REPEAT_CLOUD)
        results[n] = {
            "mean_ms": mean, "std_ms": std,
            "model": model_label,
            "random": mode == "random",
        }

    return results


# ── E2E estimation ────────────────────────────────────────────────────────────

def e2e_estimate(
    t_edge: float,
    t_prumerge: float,
    quant: dict,
    cloud: dict,
    scenario: tuple,
) -> dict:
    n, bits, bw = scenario
    t_q   = (quant.get((n, bits)) or {}).get("mean_ms", 0.0) or 0.0
    t_tx  = calc_t_tx(n, bits, bw)
    t_cl  = (cloud.get(n) or {}).get("mean_ms", 0.0) or 0.0
    total = t_edge + t_prumerge + t_q + t_tx + t_cl

    parts = {
        "T_edge":     t_edge,
        "T_prumerge": t_prumerge,
        "T_quant":    t_q,
        "T_tx":       t_tx,
        "T_cloud":    t_cl,
    }
    bottleneck = max(parts, key=lambda k: parts[k])
    pct = {k: (v / total * 100 if total > 0 else 0.0) for k, v in parts.items()}
    return {"scenario": scenario, "parts": parts, "total_ms": total,
            "bottleneck": bottleneck, "pct": pct}


# ── Output ────────────────────────────────────────────────────────────────────

SEP  = "─" * 62
SEP2 = "═" * 62

def hdr(title: str):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def row(label: str, mean: Optional[float], std: Optional[float], note: str = ""):
    if mean is None:
        print(f"  {label:<32} FAILED  {note}")
    else:
        print(f"  {label:<32} {mean:8.3f} ± {std:6.3f} ms  {note}")


def print_results(
    device_name: str, sys_info: dict,
    t_edge_res: dict, t_vit_pm_res: dict,
    quant: dict, tx_table: list, cloud: dict,
):
    # ── Device ──
    hdr("裝置資訊")
    print(f"  裝置名稱    : {device_name}")
    print(f"  GPU 型號    : {sys_info['gpu']}")
    print(f"  CUDA 版本   : {sys_info['cuda_version']}")
    print(f"  RAM         : {sys_info['ram_gb']} GB")
    print(f"  Python      : {sys_info['python']}")
    print(f"  PyTorch     : {sys_info['torch']}")

    # ── T_edge ──
    hdr("T_edge  （CLIPVisionModel，純 ViT 推論）")
    fallback_note = "[fallback: ViT-B]" if t_edge_res.get("fallback") else ""
    row(f"T_edge ({t_edge_res['model'].split('/')[-1]})",
        t_edge_res["mean_ms"], t_edge_res["std_ms"], fallback_note)

    # ── T_vit_prumerge ──
    hdr("T_vit_prumerge  （CLIPVisionTower：ViT + PruMerge + TurboQuant）")
    n_out = t_vit_pm_res.get("n_out")
    note  = f"→ {n_out} tokens" if n_out else ""
    row("T_vit_prumerge（總計）", t_vit_pm_res["mean_ms"], t_vit_pm_res["std_ms"], note)

    bd = t_vit_pm_res.get("breakdown", {})
    if bd:
        print()
        for label, key in [
            ("  ├ Stage 1  ViT forward",       "t_vit_ms"),
            ("  ├ Stage 2  Attn+TopK+Gather",  "t_attn_topk_ms"),
            ("  ├ Stage 3  Merge loop",         "t_merge_loop_ms"),
            ("  └ Stage 4  TurboQuant（推算）", None),
        ]:
            if key:
                d = bd[key]
                row(label, d["mean_ms"], d["std_ms"])
            else:
                tq = bd.get("t_quant_derived_ms", 0.0)
                print(f"  {label:<32} {tq:8.3f} ms  (total − stages 1-3)")

    t_edge_val = t_edge_res["mean_ms"] or 0.0
    t_vit_pm   = t_vit_pm_res["mean_ms"]
    # T_prumerge for E2E = Stage2 + Stage3 (attention + merge loop)
    if bd:
        t_prumerge_derived = (bd.get("t_attn_topk_ms", {}).get("mean_ms", 0.0) +
                              bd.get("t_merge_loop_ms", {}).get("mean_ms", 0.0))
    elif t_vit_pm:
        t_q_1bit = (quant.get((n_out or 35, 1)) or {}).get("mean_ms", 0.0) or 0.0
        t_prumerge_derived = max(t_vit_pm - t_edge_val - t_q_1bit, 0.0)
    else:
        t_prumerge_derived = 0.0

    # ── T_quant ──
    hdr(f"T_quant  （MSECompressor.compress + decompress，真實 TurboQuant）")
    if not quant:
        print("  [FAILED] MSECompressor import 失敗")
    else:
        for bits in BIT_VALUES:
            print(f"\n  {bits}bit:")
            for n in N_VALUES:
                d = quant.get((n, bits), {})
                row(f"  N={n}", d.get("mean_ms"), d.get("std_ms"))

    # ── T_tx ──
    hdr("T_tx  （傳輸時間，公式計算）")
    print(f"  {'N':>4}  {'bits':>4}  {'BW(Mbps)':>9}  {'T_tx(ms)':>10}")
    print(f"  {SEP}")
    for r in tx_table:
        print(f"  {r['N']:>4}  {r['bits']:>4}  {r['bw_mbps']:>9.1f}  {r['t_tx_ms']:>10.4f}")

    # ── T_cloud ──
    hdr(f"T_cloud  （Cloud LLM prefill）  [{cloud[N_VALUES[0]]['model']}]")
    if cloud[N_VALUES[0]].get("random"):
        print("  [注意] 使用隨機初始化的 LLaMA-7B 架構（無預訓練權重）")
    for n in N_VALUES:
        d = cloud[n]
        row(f"N={n}", d["mean_ms"], d["std_ms"])

    # ── E2E ──
    hdr("端到端延遲估算")
    t_pm = t_prumerge_derived

    scenarios = [
        ("最佳情況", (15, 1, 20)),
        ("典型情況", (35, 4,  5)),
        ("最差情況", (75, 4,  1)),
    ]
    e2e_results = []
    for label, sc in scenarios:
        r = e2e_estimate(t_edge_val, t_pm, quant, cloud, sc)
        e2e_results.append((label, r))
        n, bits, bw = sc
        print(f"\n  [{label}]  N={n}, B={bits}bit, S={bw}Mbps")
        print(f"  {'總延遲':<22} {r['total_ms']:8.3f} ms")
        for k, v in r["parts"].items():
            print(f"    {k:<16} {v:8.3f} ms  ({r['pct'][k]:5.1f}%)")
        print(f"  >> 瓶頸：{r['bottleneck']}")

    # ── Bottleneck bar ──
    hdr("瓶頸分析（典型情況，N=35 B=4bit S=5Mbps）")
    _, typ_r = e2e_results[1]
    for k, v in typ_r["parts"].items():
        bar = "█" * int(typ_r["pct"][k] / 2)
        print(f"  {k:<16} {typ_r['pct'][k]:5.1f}%  {bar}")
    print(f"\n  主要瓶頸：{typ_r['bottleneck']}")
    print()


# ── CSV ──────────────────────────────────────────────────────────────────────

def export_csv(
    device_name: str,
    t_edge_res: dict, t_vit_pm_res: dict,
    quant: dict, tx_table: list, cloud: dict,
):
    safe    = device_name.replace(" ", "_").replace("/", "-")
    fname   = f"{safe}_latency_results.csv"
    rows    = []

    def add(component, N, bits, bw, mean, std, note=""):
        rows.append({"component": component, "N": N, "bits": bits,
                      "bw_mbps": bw, "mean_ms": mean, "std_ms": std, "note": note})

    add("T_edge", "", "", "",
        t_edge_res["mean_ms"], t_edge_res["std_ms"],
        t_edge_res["model"] + (" [fallback]" if t_edge_res["fallback"] else ""))

    add("T_vit_prumerge", t_vit_pm_res.get("n_out", ""), "", "",
        t_vit_pm_res["mean_ms"], t_vit_pm_res["std_ms"],
        t_vit_pm_res.get("note", ""))

    for (n, b), d in quant.items():
        add("T_quant", n, b, "", d["mean_ms"], d["std_ms"], "MSECompressor")

    for r in tx_table:
        add("T_tx", r["N"], r["bits"], r["bw_mbps"], r["t_tx_ms"], 0.0, "calculated")

    for n, d in cloud.items():
        suffix = " [random weights]" if d.get("random") else ""
        add("T_cloud", n, "", "", d["mean_ms"], d["std_ms"], d["model"] + suffix)

    fields = ["component", "N", "bits", "bw_mbps", "mean_ms", "std_ms", "note"]
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV 已儲存：{fname}")
    return fname


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LMM Split Inference Latency Measurement")
    parser.add_argument("--device", type=str, default="Unknown Device",
                        help='裝置名稱，例如 "Jetson AGX Orin"')
    parser.add_argument("--skip-edge",    action="store_true", help="跳過 T_edge")
    parser.add_argument("--skip-prumerge", action="store_true", help="跳過 T_vit_prumerge")
    parser.add_argument("--skip-cloud",   action="store_true", help="跳過 T_cloud")
    args = parser.parse_args()

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP2)
    print("  LMM Split Inference Latency Measurement (Real Code)")
    print(SEP2)
    print(f"  裝置      : {args.device}")
    print(f"  Torch 裝置: {device}  |  GPU: {sys_info['gpu']}")

    # ── T_edge ──
    if args.skip_edge:
        t_edge_res = {"mean_ms": 0.0, "std_ms": 0.0, "model": "skipped", "fallback": False}
    else:
        print("\n[1/4] 測量 T_edge (CLIPVisionModel) …")
        t_edge_res = measure_t_edge(device)

    # ── T_vit_prumerge ──
    if args.skip_prumerge:
        t_vit_pm_res = {"mean_ms": None, "std_ms": None, "n_out": None, "note": "skipped"}
    else:
        print("\n[2/4] 測量 T_vit_prumerge (CLIPVisionTower：ViT + PruMerge + TurboQuant) …")
        t_vit_pm_res = measure_t_vit_prumerge(device)

    # ── T_quant ──
    print("\n[3/4] 測量 T_quant (MSECompressor) …")
    quant = measure_t_quant(device)

    # ── T_tx ──
    tx_table = build_t_tx_table()

    # ── T_cloud ──
    if args.skip_cloud:
        cloud = {n: {"mean_ms": 0.0, "std_ms": 0.0, "model": "skipped", "random": False}
                  for n in N_VALUES}
    else:
        print("\n[4/4] 測量 T_cloud …")
        cloud = measure_t_cloud(device)

    print_results(args.device, sys_info, t_edge_res, t_vit_pm_res, quant, tx_table, cloud)

    hdr("CSV 輸出")
    export_csv(args.device, t_edge_res, t_vit_pm_res, quant, tx_table, cloud)


if __name__ == "__main__":
    main()
