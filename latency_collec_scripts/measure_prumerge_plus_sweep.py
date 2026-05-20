#!/usr/bin/env python3
"""
measure_prumerge_plus_sweep.py
Captures ViT output once, sweeps topk_count=1..288 to time PruMerge plus body only.

Spatial sampling follows the real inference path (if_adaptive=True style):
    step_length = 576 // topk_count
    arith = arange(0, 575, max(1, step_length // 3))   ← same density as real system
    spatial = arith positions NOT already in topk idx (filtered)
    left_tokens = topk_count + len(spatial)
    output = left_tokens + 1

Output token count grows with topk_count as: N_out ≈ topk + 3*topk = 4*topk (roughly),
capped at 576. The CSV records actual N_out for each topk_count.

Usage:
    python measure_prumerge_plus_sweep.py --device "Jetson AGX Orin"
    python measure_prumerge_plus_sweep.py --device "Jetson AGX Orin" --topk-step 1
Output:
    results_pm_plus_sweep_<device>.csv  (columns: component, N, topk, mean_ms, std_ms)
    N = actual output token count
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

TOPK_MIN     = 1
TOPK_MAX     = 288   # topk=288 → output ≈ 576 tokens (fully covers plus range)
DEFAULT_STEP = 4
WARMUP       = 5
REPEAT       = 20

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

# ── Build & capture ───────────────────────────────────────────────────────────

def build_vit_tower(device):
    from transformers import CLIPVisionModel, CLIPVisionConfig
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
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
    return tower.to(device=device, dtype=torch.float16).eval()

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

# ── PruMerge plus body (fixed topk, if_adaptive=True style spatial) ───────────

def prumerge_plus_body(img_feat, k_t, q_t, topk_count, device):
    """
    PruMerge plus post-ViT body with fixed topk_count.
    Spatial supplementation mirrors real inference (if_adaptive=True):
        step_length = 576 // topk_count
        arith = arange(0, 575, max(1, step_length // 3))
        filtered = arith positions not in topk idx
    Output = topk_count + len(filtered) + 1 tokens.
    """
    from llava.model.multimodal_encoder.clip_encoder import complement_idx
    B, N, C = img_feat.shape
    topk_count = max(1, min(topk_count, N - 2))

    attn = (q_t @ k_t.transpose(-2, -1)) * C ** -0.5
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:]

    _, idx = torch.topk(cls_attn, topk_count, dim=1, largest=True)

    # Spatial supplementation — real inference (if_adaptive=True) path
    step_length = max(1, N // topk_count)
    arith_step  = max(1, step_length // 3)
    arith       = torch.arange(0, 575, arith_step, device=device)
    orig_1d     = idx.flatten()
    topk_set    = set(orig_1d.tolist())
    filtered    = torch.tensor([x.item() for x in arith if x.item() not in topk_set],
                                dtype=torch.long, device=device)
    if filtered.numel() > 0:
        idx = torch.cat((idx, filtered.unsqueeze(0)), dim=1)

    index = idx.unsqueeze(-1).expand(-1, -1, C)

    Key_wo_cls    = k_t[:, 1:]
    x_others      = torch.gather(img_feat,   dim=1, index=index)
    x_others_attn = torch.gather(cls_attn,   dim=1, index=idx)
    Key_others    = torch.gather(Key_wo_cls, dim=1, index=index)
    compl         = complement_idx(idx, N)
    non_topk      = torch.gather(img_feat,   dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_Key  = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn,   dim=1, index=compl)

    Key_others_norm   = F.normalize(Key_others,   p=2, dim=-1)
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
            sim   = torch.bmm(kn, rest_k.transpose(1, 2))
            n_cl  = min(32, rest_k.shape[1])
            _, ci = torch.topk(sim, k=n_cl, dim=2, largest=True)
            ct    = rest_x[:, ci.squeeze(0).squeeze(0), :]
            w     = rest_a[:, ci.squeeze(0).squeeze(0)].unsqueeze(-1)
            updated[b, i] = x_others[b, i] + torch.sum(ct * w, dim=1)

    extra = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    return torch.cat([updated, extra], dim=1)

# ── Sweep ─────────────────────────────────────────────────────────────────────

def sweep(img_feat, k_t, q_t, topk_values, device):
    """
    For each topk_count: time prumerge_plus_body, record actual output N.
    Returns list of {"topk": ..., "N_out": ..., "mean_ms": ..., "std_ms": ...}
    """
    results = []
    total   = len(topk_values)

    for pos, topk in enumerate(topk_values):
        # Probe actual output size once (no timing)
        with torch.no_grad():
            probe = prumerge_plus_body(img_feat, k_t, q_t, topk, device)
        n_out = probe.shape[1]

        def run(tc=topk):
            with torch.no_grad():
                prumerge_plus_body(img_feat, k_t, q_t, tc, device)

        mean, std = timed_run(run, WARMUP, REPEAT)
        results.append({"topk": topk, "N_out": n_out, "mean_ms": mean, "std_ms": std})

        if (pos + 1) % 10 == 0 or pos == 0 or pos == total - 1:
            print(f"  [{pos+1:>4}/{total}] topk={topk:<4}  N_out={n_out:<4}  "
                  f"{mean:.2f} ± {std:.2f} ms", flush=True)

    return results

# ── Print & CSV ───────────────────────────────────────────────────────────────

SEP = "═" * 62

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_summary(results):
    hdr("T_pm_plus sweep  — sample rows")
    print(f"  {'topk':<6}  {'N_out':<6}  {'mean_ms':>10}  {'std_ms':>8}")
    print(f"  {'─'*36}")
    step = max(1, len(results) // 8)
    indices = sorted(set([0] + list(range(0, len(results), step)) + [len(results)-1]))
    for i in indices:
        r = results[i]
        print(f"  {r['topk']:<6}  {r['N_out']:<6}  {r['mean_ms']:>10.3f}  {r['std_ms']:>8.3f} ms")
    print(f"\n  (full {len(results)} rows saved to CSV)")

def export_csv(device_name, results):
    safe  = device_name.replace(" ", "_").replace("/", "-")
    fname = str(Path(__file__).resolve().parent / f"results_pm_plus_sweep_{safe}.csv")
    fields = ["component", "N", "topk", "mean_ms", "std_ms"]
    rows = [{"component": "T_pm_plus", "N": r["N_out"], "topk": r["topk"],
              "mean_ms": r["mean_ms"], "std_ms": r["std_ms"]}
             for r in results]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {fname}  ({len(rows)} rows)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PruMerge plus body latency sweep over topk_count=1..288")
    parser.add_argument("--device",     type=str, default="Unknown Device")
    parser.add_argument("--topk-step",  type=int, default=DEFAULT_STEP,
                        help=f"topk_count step (default {DEFAULT_STEP}). Use 1 for all 288 values.")
    parser.add_argument("--topk-values", type=int, nargs="+", default=None,
                        help="Explicit topk_count values (overrides --topk-step)")
    args = parser.parse_args()

    if args.topk_values:
        topk_values = sorted(set(t for t in args.topk_values if TOPK_MIN <= t <= TOPK_MAX))
    else:
        topk_values = list(range(TOPK_MIN, TOPK_MAX + 1, args.topk_step))
        if TOPK_MAX not in topk_values:
            topk_values.append(TOPK_MAX)

    device   = get_device()
    sys_info = detect_system_info()

    print(SEP)
    print("  PruMerge Plus Sweep  →  results_pm_plus_sweep_<device>.csv")
    print(SEP)
    print(f"  Device     : {args.device}  |  GPU: {sys_info['gpu']}")
    print(f"  topk range : {topk_values[0]}..{topk_values[-1]}  ({len(topk_values)} values, step={args.topk_step})")
    print(f"  N_out range: approx 4*topk  (actual N recorded in CSV)")
    print(f"  Per topk   : warmup={WARMUP}, repeat={REPEAT}")

    print("\n[1/3] Building ViT Tower ...")
    tower = build_vit_tower(device)
    pv    = torch.randn(1, 3, 336, 336, device=device, dtype=torch.float16)

    print("\n[2/3] Capturing ViT features (one-time) ...")
    img_feat, k_t, q_t = capture_vit(tower, pv)
    del tower
    print(f"  img_feat: {list(img_feat.shape)},  k_t: {list(k_t.shape)},  q_t: {list(q_t.shape)}")

    print(f"\n[3/3] Sweeping PruMerge plus body ...")
    results = sweep(img_feat, k_t, q_t, topk_values, device)

    print_summary(results)

    hdr("CSV Output")
    export_csv(args.device, results)

if __name__ == "__main__":
    main()
