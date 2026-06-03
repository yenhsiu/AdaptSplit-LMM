#!/usr/bin/env python3
"""
measure_latency_prumerge_full.py — 手法②④ 用（整個 model）

【モード1：fixed】デフォルト
  ViT + PruMerge(fixed ratio) + Projector + LLM を 2 点測定
    base: reduction_ratio=1/16 → 36 tokens
    plus: reduction_ratio=1/8  → 144 tokens

【モード2：sweep  --n-step N】
  ViT を 1 回だけキャプチャ → PruMerge_body(N) + Projector + LLM を N=1..576 スイープ
  （ViT を毎回回す必要がないため高速）
  T_vit は定数として別行で CSV に記録される

Usage:
    python measure_latency_prumerge_full.py --device "Jetson AGX Orin"
    python measure_latency_prumerge_full.py --device "Jetson AGX Orin" --n-step 8
    python measure_latency_prumerge_full.py --device "Jetson AGX Orin" --n-step 1   # all 576
Output:
    results_latency_pm_full_<device>.csv
    columns: component, N, mean_ms, std_ms
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

TEXT_TOKENS    = 40
VIT_DIM        = 1024
PROJ_DIM       = 4096
N_MIN          = 1
N_MAX          = 576
WARMUP         = 10
REPEAT         = 20
REPEAT_SWEEP   = 10   # fewer reps for sweep (many N values)
REPEAT_VIT     = 100  # T_vit separate measurement

BASE_RATIO     = 1 / 16  # → 36 tokens
PLUS_RATIO     = 1 / 8   # → 144 tokens (if_adaptive=False)

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

def build_tower(device):
    from transformers import CLIPVisionModel, CLIPVisionConfig
    from llava.model.multimodal_encoder.clip_encoder_t import CLIPVisionTower
    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        image_size=336, patch_size=14, projection_dim=768,
    )
    tower = CLIPVisionTower.__new__(CLIPVisionTower)
    nn.Module.__init__(tower)
    tower.is_loaded = False
    tower.vision_tower_name = "openai/clip-vit-large-patch14-336"
    tower.select_layer = -2
    tower.select_feature = "patch"
    tower.total_tokens = 0
    tower.image_processor = None
    tower.vision_tower = CLIPVisionModel(config)
    tower.vision_tower.requires_grad_(False)
    tower.is_loaded = True
    tower = tower.to(device=device, dtype=torch.float16).eval()
    print("  Built CLIPVisionTower (ViT-L + PruMerge, random weights).", flush=True)
    return tower

def build_projector(device):
    proj = nn.Sequential(
        nn.Linear(VIT_DIM, PROJ_DIM),
        nn.GELU(),
        nn.Linear(PROJ_DIM, PROJ_DIM),
    ).to(device=device, dtype=torch.float16).eval()
    print("  Built MLP Projector (1024→4096).", flush=True)
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

# ── ViT capture (for sweep) ───────────────────────────────────────────────────

def capture_vit(tower, pv):
    """Run ViT once with hooks; return (img_feat, k, q) cloned."""
    captured = {}
    def hk(m, i, o): captured['k'] = o.detach().clone()
    def hq(m, i, o): captured['q'] = o.detach().clone()
    h1 = tower.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hk)
    h2 = tower.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hq)
    with torch.no_grad():
        out = tower.vision_tower(pv, output_hidden_states=True)
    h1.remove(); h2.remove()
    img_feat = out.hidden_states[-2][:, 1:].to(pv.dtype).clone()
    return img_feat, captured['k'], captured['q']

# ── PruMerge base body (fixed ratio, no outlier_detection) ───────────────────

def prumerge_base_body_fixed(img_feat, k_t, q_t, reduction_ratio):
    """PruMerge base post-ViT operations with a fixed reduction_ratio."""
    from llava.model.multimodal_encoder.clip_encoder import complement_idx

    B, N, C = img_feat.shape
    n_keep = max(1, min(int(N * reduction_ratio), N - 1))

    attn = (q_t @ k_t.transpose(-2, -1)) * C ** -0.5
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:]

    _, idx = torch.topk(cls_attn, n_keep, dim=1, largest=True)
    index = idx.unsqueeze(-1).expand(-1, -1, C)

    Key_wo_cls    = k_t[:, 1:]
    x_others      = torch.gather(img_feat, dim=1, index=index)
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
    Key_others    = torch.gather(Key_wo_cls, dim=1, index=index)
    compl         = complement_idx(idx, N)
    non_topk      = torch.gather(img_feat, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_Key  = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

    Key_others_norm   = F.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

    B2, left_tokens, _ = x_others.size()
    updated = torch.zeros_like(x_others)

    for b in range(B2):
        for i in range(left_tokens):
            kn     = Key_others_norm[b, i].unsqueeze(0).unsqueeze(0)
            rest_k = torch.cat([Key_others_norm[b, :i].unsqueeze(0),
                                 Key_others_norm[b, i+1:].unsqueeze(0),
                                 non_topk_Key_norm[b].unsqueeze(0)], dim=1)
            rest_x = torch.cat([x_others[b, :i].unsqueeze(0),
                                 x_others[b, i+1:].unsqueeze(0),
                                 non_topk[b].unsqueeze(0)], dim=1)
            rest_a = torch.cat([x_others_attn[b, :i].unsqueeze(0),
                                 x_others_attn[b, i+1:].unsqueeze(0),
                                 non_topk_attn[b].unsqueeze(0)], dim=1)
            sim    = torch.bmm(kn, rest_k.transpose(1, 2))
            n_cl   = min(32, rest_k.shape[1])
            _, ci  = torch.topk(sim, k=n_cl, dim=2, largest=True)
            ct     = rest_x[:, ci.squeeze(0).squeeze(0), :]
            w      = rest_a[:, ci.squeeze(0).squeeze(0)].unsqueeze(-1)
            updated[b, i] = torch.sum(ct * w, dim=1) + x_others[b, i]

    extra = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    return torch.cat([updated, extra], dim=1)

# ── Mode 1: fixed (base + plus) ───────────────────────────────────────────────

def measure_fixed(variant, tower, projector, llm, pv, text_embeds):
    log_attr = "latency_log" if variant == "base" else "latency_log_plus"
    ratio    = BASE_RATIO if variant == "base" else PLUS_RATIO

    setattr(tower, log_attr, [])
    with torch.no_grad():
        if variant == "base":
            probe = tower.token_prune_merge_advanced(pv, if_adaptive=False, reduction_ratio=ratio)
        else:
            probe = tower.token_prune_merge_advanced_plus(pv, if_adaptive=False, reduction_ratio=ratio)
    n_out = probe.shape[1]
    print(f"  PruMerge {variant} output: {n_out} tokens", flush=True)

    setattr(tower, log_attr, [])

    def run():
        with torch.no_grad():
            if variant == "base":
                img_feat = tower.token_prune_merge_advanced(pv, if_adaptive=False, reduction_ratio=ratio)
            else:
                img_feat = tower.token_prune_merge_advanced_plus(pv, if_adaptive=False, reduction_ratio=ratio)
            proj_out = projector(img_feat)
            inputs   = torch.cat([proj_out, text_embeds], dim=1)
            llm(inputs_embeds=inputs, use_cache=False)

    print(f"  Warm-up {WARMUP} + measure {REPEAT} runs ...", flush=True)
    mean, std = timed_run(run, WARMUP, REPEAT)

    log = getattr(tower, log_attr, [])
    breakdown = {k: sum(e[k] for e in log) / len(log) for k in
                 ("t_vit_ms", "t_attn_topk_ms", "t_merge_loop_ms") if log}

    return {"mean_ms": mean, "std_ms": std, "n_out": n_out, "breakdown": breakdown}

# ── Mode 2: sweep ─────────────────────────────────────────────────────────────

def measure_sweep(tower, projector, llm, device, pv, text_embeds, n_values):
    """
    ViT を 1 回キャプチャ → PruMerge_base_body(N) + Projector + LLM をスイープ。
    T_vit は別途測定して CSV に記録。
    """
    # T_vit
    print("  Measuring T_vit (ViT only) ...", flush=True)
    def vit_run():
        with torch.no_grad():
            tower.vision_tower(pv, output_hidden_states=True)
    t_vit_mean, t_vit_std = timed_run(vit_run, WARMUP, REPEAT_VIT)
    print(f"  T_vit = {t_vit_mean:.3f} ± {t_vit_std:.3f} ms", flush=True)

    # Capture ViT output once
    print("  Capturing ViT features ...", flush=True)
    img_feat, k_t, q_t = capture_vit(tower, pv)

    # Sweep: PruMerge_body(N) + Projector + LLM
    print(f"  Sweeping PruMerge_body + Proj + LLM for {len(n_values)} N values ...", flush=True)
    sweep_results = {}
    total = len(n_values)

    for idx, n in enumerate(n_values):
        ratio = n / N_MAX   # reduction_ratio so topk keeps n tokens

        def run(r=ratio):
            with torch.no_grad():
                out      = prumerge_base_body_fixed(img_feat, k_t, q_t, r)
                proj_out = projector(out)
                inputs   = torch.cat([proj_out, text_embeds], dim=1)
                llm(inputs_embeds=inputs, use_cache=False)

        mean, std = timed_run(run, WARMUP, REPEAT_SWEEP)
        sweep_results[n] = {"mean_ms": mean, "std_ms": std}

        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"  [{idx+1:>4}/{total}] N={n:<4}  {mean:.1f} ± {std:.1f} ms", flush=True)

    return t_vit_mean, t_vit_std, sweep_results

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_fixed(r_base, r_plus):
    def show(label, r):
        if r["mean_ms"] is None:
            print("  skipped")
            return
        print(f"  Total : {r['mean_ms']:.3f} ± {r['std_ms']:.3f} ms  ({r['n_out']} tokens)")
        bd = r.get("breakdown", {})
        if bd:
            print(f"    ├ ViT forward  : {bd.get('t_vit_ms', 0):.3f} ms")
            print(f"    ├ Attn+TopK    : {bd.get('t_attn_topk_ms', 0):.3f} ms")
            print(f"    └ Merge loop   : {bd.get('t_merge_loop_ms', 0):.3f} ms")
            rest = r['mean_ms'] - sum(bd.values())
            print(f"      Proj+LLM     : {rest:.3f} ms  (derived)")
    hdr(f"T_latency_prumerge_base  (ViT+PruMerge_base+Proj+LLM, ratio=1/16→{r_base.get('n_out','?')} tokens)")
    show("base", r_base)
    hdr(f"T_latency_prumerge_plus  (ViT+PruMerge_plus+Proj+LLM, ratio=1/8→{r_plus.get('n_out','?')} tokens)")
    show("plus", r_plus)

def print_sweep(t_vit_mean, t_vit_std, sweep_results, n_values):
    hdr("Sweep: PruMerge_base_body + Proj + LLM  (T_vit is constant, add separately)")
    print(f"  T_vit  : {t_vit_mean:.3f} ± {t_vit_std:.3f} ms  (add to each row for T_total)")
    landmarks = sorted(set([n_values[0]] + [n for n in [36, 145, 576] if n in sweep_results] + [n_values[-1]]))
    print(f"\n  {'N':<6}  {'T_pm_proj_llm':>14}  {'std':>8}")
    print(f"  {'─'*34}")
    for n in landmarks:
        d = sweep_results[n]
        print(f"  N={n:<4}  {d['mean_ms']:>14.3f}  {d['std_ms']:>8.3f} ms")
    print(f"\n  (full {len(sweep_results)} rows saved to CSV)")

def export_csv(device_name, r_base, r_plus, t_vit_mean, t_vit_std, sweep_results):
    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / f"results_latency_pm_full_{safe}.csv")
    fields = ["component", "N", "mean_ms", "std_ms"]
    rows = []

    # Fixed measurements
    if r_base["mean_ms"] is not None:
        rows.append({"component": "T_latency_prumerge_base", "N": r_base.get("n_out", "-"),
                     "mean_ms": r_base["mean_ms"], "std_ms": r_base["std_ms"]})
    if r_plus["mean_ms"] is not None:
        rows.append({"component": "T_latency_prumerge_plus", "N": r_plus.get("n_out", "-"),
                     "mean_ms": r_plus["mean_ms"], "std_ms": r_plus["std_ms"]})

    # Sweep: T_vit constant + per-N pm_proj_llm
    if t_vit_mean is not None:
        rows.append({"component": "T_vit", "N": N_MAX,
                     "mean_ms": t_vit_mean, "std_ms": t_vit_std})
    for n in sorted(sweep_results):
        d = sweep_results[n]
        rows.append({"component": "T_pm_proj_llm_sweep", "N": n,
                     "mean_ms": d["mean_ms"], "std_ms": d["std_ms"]})

    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {fname}  ({len(rows)} rows)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline latency: ViT + PruMerge + Projector + LLM")
    parser.add_argument("--device", type=str, default="Unknown Device")
    parser.add_argument("--skip-base",  action="store_true")
    parser.add_argument("--skip-plus",  action="store_true")
    parser.add_argument("--n-step", type=int, default=None,
                        help="Enable sweep mode: step size for N=1..576 (e.g. 8). "
                             "Omit to run fixed-point mode only.")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="Explicit N values for sweep (overrides --n-step)")
    args = parser.parse_args()

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  measure_latency_prumerge_full.py  →  results_latency_pm_full_<device>.csv")
    print(SEP)
    print(f"  Device: {args.device}  |  Torch: {device}  |  GPU: {sys_info['gpu']}")

    print("\n[1/4] Building models ...")
    tower     = build_tower(device)
    projector = build_projector(device)
    llm       = build_llm(device)

    pv          = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)
    text_embeds = torch.randn(1, TEXT_TOKENS, PROJ_DIM, device=device, dtype=torch.float16)

    # ── Mode 1: fixed point ──
    r_base = {"mean_ms": None, "std_ms": None, "n_out": None}
    r_plus = {"mean_ms": None, "std_ms": None, "n_out": None}

    if not args.skip_base:
        print("\n[2/4] T_latency_prumerge_base (fixed, ratio=1/16) ...")
        r_base = measure_fixed("base", tower, projector, llm, pv, text_embeds)

    if not args.skip_plus:
        print("\n[3/4] T_latency_prumerge_plus (fixed, ratio=1/8) ...")
        r_plus = measure_fixed("plus", tower, projector, llm, pv, text_embeds)

    hdr("Fixed-point Results")
    print_fixed(r_base, r_plus)

    # ── Mode 2: sweep ──
    t_vit_mean, t_vit_std = None, None
    sweep_results = {}

    if args.n_step is not None or args.n_values is not None:
        if args.n_values:
            n_values = sorted(set(n for n in args.n_values if N_MIN <= n <= N_MAX))
        else:
            n_values = list(range(N_MIN, N_MAX + 1, args.n_step))
            if N_MAX not in n_values:
                n_values.append(N_MAX)

        print(f"\n[4/4] Sweep mode: N={n_values[0]}..{n_values[-1]} ({len(n_values)} values) ...")
        t_vit_mean, t_vit_std, sweep_results = measure_sweep(
            tower, projector, llm, device, pv, text_embeds, n_values)
        print_sweep(t_vit_mean, t_vit_std, sweep_results, n_values)
    else:
        print("\n[4/4] Sweep skipped (add --n-step 8 to enable)")

    hdr("CSV Output")
    export_csv(args.device, r_base, r_plus, t_vit_mean, t_vit_std, sweep_results)

if __name__ == "__main__":
    main()
