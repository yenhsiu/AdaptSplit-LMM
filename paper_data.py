"""
Shared data for the paper figures (plot_vqav2_figures.py / plot_paper_figures.py /
plot_figure4.py) -- single source of truth so the three scripts can't drift apart.

Grid: 8 anchor points, 25,600-589,824 bits (spec calls for 9; only 8 exist in
results/vqa_anchore/pipeline_vqav2/ as of 2026-07-28 -- confirmed with the user to
proceed on these 8 until the 9th point is supplied/measured).

Naming: "All B=1/2/4" renamed to SP-Q1/SP-Q2/SP-Q4 per user request (2026-07-28).
"VQAv2-opt" (the proposed method) is unchanged.
"""

BUDGETS = [25600, 41000, 66000, 106000, 170000, 273000, 439000, 589824]

METHODS = ["SP-Q1", "SP-Q2", "SP-Q4", "VQAv2-opt"]

# ============================================================
# Per-anchor benchmark scores (verified against
# results/vqa_anchore/pipeline_vqav2/*.txt, 2026-07-18)
# ============================================================
DATA = {
    "MME": {
        "SP-Q1": [1645.74, 1658.91, 1685.53, 1646.74, 1639.98, 1623.84, 1597.21, 1585.45],
        "SP-Q2": [1482.47, 1625.82, 1662.35, 1690.78, 1709.51, 1695.91, 1647.35, 1669.42],
        "SP-Q4": [1121.24, 1374.86, 1574.83, 1656.04, 1695.03, 1711.39, 1667.64, 1659.94],
        "VQAv2-opt": [1645.74, 1665.11, 1695.31, 1675.55, 1691.35, 1656.85, 1619.11, 1615.93],
    },
    "POPE": {
        "SP-Q1": [0.7323, 0.7767, 0.7917, 0.8043, 0.8028, 0.8110, 0.8435, 0.8488],
        "SP-Q2": [0.6330, 0.7126, 0.7578, 0.7868, 0.8033, 0.8123, 0.8146, 0.8130],
        "SP-Q4": [0.4337, 0.5954, 0.6910, 0.7492, 0.7834, 0.7933, 0.8143, 0.8130],
        "VQAv2-opt": [0.7323, 0.7746, 0.7810, 0.7912, 0.8041, 0.8070, 0.8399, 0.8459],
    },
    "TextVQA": {
        "SP-Q1": [53.25, 54.35, 54.59, 54.34, 54.63, 53.73, 53.21, 52.92],
        "SP-Q2": [50.82, 53.11, 54.71, 55.51, 55.51, 55.61, 55.74, 55.23],
        "SP-Q4": [46.04, 50.00, 52.56, 54.03, 55.65, 56.25, 56.17, 56.01],
        "VQAv2-opt": [53.25, 54.39, 55.37, 55.51, 55.54, 56.01, 54.46, 54.88],
    },
    "MMBench": {
        "SP-Q1": [68.04, 69.41, 70.09, 69.89, 69.73, 70.94, 71.19, 70.23],
        "SP-Q2": [63.97, 68.68, 69.82, 70.44, 70.39, 71.33, 71.03, 72.47],
        "SP-Q4": [45.46, 61.30, 67.67, 69.11, 70.60, 70.82, 71.37, 71.01],
        "VQAv2-opt": [68.04, 69.59, 69.98, 70.41, 70.80, 70.82, 71.56, 71.56],
    },
}

COLORS = {"SP-Q1": "#2a78d6", "SP-Q2": "#eb6834", "SP-Q4": "#1baf7a", "VQAv2-opt": "#e34948"}
STYLES = {"SP-Q1": "--", "SP-Q2": "-.", "SP-Q4": ":", "VQAv2-opt": "-"}
WIDTHS = {"SP-Q1": 1.5, "SP-Q2": 1.5, "SP-Q4": 1.5, "VQAv2-opt": 2.5}

# Display-only legend labels (data dict keys above stay "VQAv2-opt" everywhere
# else so N_BY_METHOD / T_CLOUD_BY_METHOD / DATA don't need to change).
# Rename here if "Proposed (Ours)" isn't the label you want.
LABELS = {"SP-Q1": "SP-Q1", "SP-Q2": "SP-Q2", "SP-Q4": "SP-Q4", "VQAv2-opt": "Proposed (Ours)"}

# Figure 3 ONLY, fixed/non-adaptive variant: SP-Q1/Q2/Q4 always send N=576 at a
# fixed bit-depth, ignoring bandwidth/T_deadline entirely -- a different
# experiment from the budget-matched SP-Q1/Q2/Q4 swept in Figures 1/2
# (paper_data.N_BY_METHOD), so it gets a distinct label -- decided with the
# user 2026-07-29 -- to avoid implying it's the same setup. Figure 3 ALSO
# plots the budget-matched SP-Q1/Q2/Q4 (dynamic snap-down over their own
# 8-anchor family, same definition as Figures 1/2) using LABELS above, so
# Figure 3 shows both variants side by side. No static counterpart for the
# proposed method -- it only has the adaptive form.
FIG3_STATIC_LABELS = {"SP-Q1": "Static-Q1", "SP-Q2": "Static-Q2", "SP-Q4": "Static-Q4"}

# ============================================================
# Per-method token count N at each budget.
# SP-Q1/Q2/Q4: the "anchors" field of results/search_vqav2_*.json (the pure
# single-bit-depth seed configs TPE search started from -- e.g. SP-Q2 at
# budget=25,600 is (n4=0, n2=12, n1=1): 12 tokens at 2-bit plus 1 leftover
# 1-bit token to hit the exact budget).
# VQAv2-opt: the best config found by the search ("results" field of the same files).
# ============================================================
N_BY_METHOD = {
    "SP-Q1":     [25, 40, 64, 103, 166, 266, 428, 576],
    "SP-Q2":     [13, 20, 32, 52, 83, 133, 214, 288],
    "SP-Q4":     [7, 10, 16, 28, 43, 68, 107, 144],
    "VQAv2-opt": [25, 39, 43, 68, 139, 154, 392, 421],
}

# ============================================================
# T_cloud(N) in ms, looked up per-method (not shared across methods -- SP-Q1 and
# SP-Q4 use very different N at the same budget, so they have different cloud
# compute time). Source: latency_collec_scripts/final_lat/results_cloud.csv,
# a direct N=1..576 sweep measured on the current cloud model/hardware.
# ============================================================
T_CLOUD_BY_METHOD = {
    "SP-Q1":     [21.39, 21.43, 22.40, 26.68, 31.69, 39.83, 54.63, 64.96],
    "SP-Q2":     [21.13, 20.98, 21.40, 22.17, 23.39, 28.24, 31.67, 40.01],
    "SP-Q4":     [21.10, 21.18, 21.01, 21.32, 21.93, 22.65, 26.76, 27.48],
    "VQAv2-opt": [21.39, 21.43, 21.93, 22.65, 27.70, 31.29, 51.97, 56.17],
}

T_EDGE = 231
T_DEADLINE = 400

# ============================================================
# Figure 3 fixed baselines (2026-07-29): SP-Q1/Q2/Q4 there are NOT the dynamic
# 8-anchor families above -- they're a single fixed strategy (no PruMerge, no
# bandwidth-adaptation) so the comparison against the proposed method's dynamic
# snap-down is "adapts nothing" vs "adapts both N and bit-depth". Fixed at
# N=576 (send every token, no pruning) so the reference point is an
# already-measured anchor (T_CLOUD_BY_METHOD["SP-Q1"][7] == T_CLOUD_N576) rather
# than an arbitrary pick from the 8-point grid.
# ============================================================
N_FIXED_BASELINE = 576
TOKEN_DIM = 1024
T_CLOUD_N576 = 64.96
BIT_BY_METHOD = {"SP-Q1": 1, "SP-Q2": 2, "SP-Q4": 4}

# ============================================================
# FP16 (no-quant, N=576) baselines for normalization.
# MME/POPE/TextVQA carried over from the OLDER MME-objective calibration
# (results/exp_results_summary.md, 2026-06-25) -- re-verify against the CURRENT
# pipeline before trusting normalized numbers in a final figure. MMBench baseline
# was not found anywhere in the repo and is left as None.
# ============================================================
BASELINE_FP16 = {
    "MME": 1861.4314,
    "POPE": 0.8577,
    "TextVQA": 55.71,
    "MMBench": None,  # TODO: run `bash validate.sh --dataset mmbench --baseline`
}
