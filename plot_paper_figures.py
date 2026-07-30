"""
Figure 3: Bandwidth vs End-to-End Latency.

Computation only -- no new inference. Up to three tiers of comparison (decided
with the user 2026-07-29), to separate "adaptation itself helps" from "mixed-
precision search helps on top of adaptation":

  1. Static-Q1/Q2/Q4  -- always drawn. No PruMerge, no bandwidth-adaptation:
     always send all N=576 tokens at a fixed bit-depth (1/2/4), ignoring
     T_deadline entirely.
  2. SP-Q1/Q2/Q4      -- opt-in via show_spq=True / --show-spq (default OFF:
     adds no discriminating information for this figure's claim -- see below).
     SAME definition as Figures 1/2 (uniform bit-depth, N tuned per budget):
     here evaluated via "dynamic snap-down" over each method's own 8-anchor
     (budget, N, T_cloud(N)) LUT family -- at each bandwidth S, pick the
     largest budget whose T_total <= T_deadline, else the smallest/cheapest
     anchor. Adapts N, but NOT the quantization strategy (still uniform
     bit-depth).
  3. Proposed (Ours)  -- always drawn. Same dynamic snap-down, but over the
     searched mixed-precision LUT (n4/n2/n1 per anchor) instead of a uniform
     bit-depth. Adapts both N and per-token bit-depth.

Tier 2 defaults to hidden because SP-Q1/Q2/Q4 and Proposed both avoid deadline
violations via the same snap-down mechanism -- only the static tier fails to --
so tier 2 doesn't sharpen this figure's "adaptation matters" claim. Proposed's
actual advantage (better accuracy at matched budget/latency) lives in Figures
1/2, not here; adding tier 2 back mainly helps preempt a reviewer asking why
SP-Q1/Q2/Q4 vanished between figures.

    Static:       T_total(S) = T_edge + T_cloud(576) + (576*1024*B)/(S*1000)
    SP-Q / Proposed: T_total(S) = T_edge + T_cloud(N) + budget/(S*1000), with
                      N/budget chosen per-S by dynamic snap-down over that
                      method's own 8-anchor LUT

N=576 chosen for the static baselines (not an arbitrary point from the
8-anchor grid) because it's already a measured anchor -- paper_data.T_CLOUD_N576
equals T_CLOUD_BY_METHOD["SP-Q1"][7], the same N=576/B=1 measurement used
elsewhere.

Common settings:
    T_edge      = 231 ms (fixed, measured)
    T_deadline  = 400 ms

Grid: 8 anchor points, 25,600-589,824 bits (spec calls for 9; only 8 exist in
results/vqa_anchore/pipeline_vqav2/ as of 2026-07-28 -- confirmed with the user
to proceed on these 8 until the 9th point is supplied/measured).

Legend labels: SP-Q1/Q2/Q4 and Proposed (Ours) reuse paper_data.LABELS (same
definition as Figures 1/2); the static variants use paper_data.FIG3_STATIC_LABELS
(Static-Q1/Q2/Q4) since they're a different experiment, not just a renamed
series -- decided with the user 2026-07-29.

Known limitation: the static baselines transmit N=576 unconditionally, so at
low bandwidth they blow well past T_deadline (that's the point being made --
a fixed strategy has no way to back off). SP-Q1/Q2/Q4 and Proposed can both
drop below T_deadline at low S by snapping down to a cheaper anchor; only
Proposed can also trade quantization precision per token.

Usage:
    python plot_paper_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from paper_data import (BUDGETS, METHODS, COLORS, LABELS, FIG3_STATIC_LABELS,
                         T_CLOUD_BY_METHOD, T_EDGE, T_DEADLINE,
                         N_FIXED_BASELINE, TOKEN_DIM, T_CLOUD_N576, BIT_BY_METHOD)

PROPOSED = "VQAv2-opt"
STATIC_METHODS = [m for m in METHODS if m != PROPOSED]  # SP-Q1/Q2/Q4 only


def t_total_dynamic(method: str, idx: int, s_mbps: float) -> float:
    return T_EDGE + T_CLOUD_BY_METHOD[method][idx] + BUDGETS[idx] / (s_mbps * 1000)


def dynamic_snap_down(method: str, s_mbps: float) -> float:
    """Largest budget in this method's own LUT whose T_total <= T_deadline; else the smallest (cheapest) anchor."""
    totals = [t_total_dynamic(method, i, s_mbps) for i in range(len(BUDGETS))]
    feasible = [i for i, t in enumerate(totals) if t <= T_DEADLINE]
    idx = max(feasible) if feasible else 0
    return totals[idx]


def static_total(method: str, s_mbps: float) -> float:
    """Static-Q1/Q2/Q4: always send all N_FIXED_BASELINE tokens at a fixed bit-depth, regardless of S or T_deadline."""
    budget = N_FIXED_BASELINE * TOKEN_DIM * BIT_BY_METHOD[method]
    return T_EDGE + T_CLOUD_N576 + budget / (s_mbps * 1000)


def plot_figure3(out_path="figure3_bandwidth_vs_latency.png", s_min=0.1, s_max=20,
                  n_points=500, show_spq=False):
    """show_spq=False (default): only Static-Q1/Q2/Q4 + Proposed are drawn --
    SP-Q1/Q2/Q4's dynamic snap-down tier adds no discriminating information
    over Proposed for the "does adaptation matter" claim this figure makes
    (both avoid deadline violations; only the static tier fails to), so it's
    opt-in clutter. Pass show_spq=True to bring it back as a middle tier."""
    s_grid = np.geomspace(s_min, s_max, n_points)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Tier 1: static, no adaptation at all -- thin dashed
    for method in STATIC_METHODS:
        ys = [static_total(method, s) for s in s_grid]
        ax.plot(s_grid, ys, label=FIG3_STATIC_LABELS[method], color=COLORS[method],
                 linewidth=1.3, linestyle='--', alpha=0.6, zorder=2)

    # Tier 2 (opt-in): SP-Q1/Q2/Q4, dynamic snap-down but uniform bit-depth -- thin solid
    if show_spq:
        for method in STATIC_METHODS:
            ys = [dynamic_snap_down(method, s) for s in s_grid]
            ax.plot(s_grid, ys, label=LABELS[method], color=COLORS[method],
                     linewidth=1.6, linestyle='-', zorder=3)

    # Tier 3: proposed, dynamic snap-down over the mixed-precision LUT -- thick solid
    ys = [dynamic_snap_down(PROPOSED, s) for s in s_grid]
    ax.plot(s_grid, ys, label=LABELS[PROPOSED], color=COLORS[PROPOSED],
             linewidth=2.5, linestyle='-', zorder=5)

    ax.axhline(y=T_DEADLINE, color="black", linewidth=1.3, linestyle=":",
               label=f"T_deadline = {T_DEADLINE} ms")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(s_min, s_max)
    ax.set_xlabel('Bandwidth S (Mbps, log scale)')
    ax.set_ylabel('End-to-End Latency T_total (ms, log scale)')
    ax.set_title('Figure 3: Bandwidth vs End-to-End Latency')
    ax.legend(fontsize=7.5, loc='upper right', ncol=1)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-spq", action="store_true",
                         help="Also draw the SP-Q1/Q2/Q4 dynamic snap-down tier (default: off)")
    args = parser.parse_args()
    plot_figure3(show_spq=args.show_spq)
