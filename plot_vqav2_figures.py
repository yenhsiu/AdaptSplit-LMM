"""
Figure 1: Accuracy vs Transmission Budget (log scale x-axis), one panel per benchmark
Figure 2: Accuracy vs Latency (T_total at a fixed uplink bandwidth S), one panel per benchmark

Data verified against results/vqa_anchore/pipeline_vqav2/{MME,POPE,TEXTVQA,MMBENCH}_*.txt
-- every value in paper_data.py matches the corresponding table exactly.

Per-benchmark grid (not a normalized cross-benchmark average) per user request
2026-07-28. Labels renamed All B=1/2/4 -> SP-Q1/SP-Q2/SP-Q4 per user request.

Latency in Figure 2 uses each method's OWN T_cloud(N) (SP-Q1/Q2/Q4 use very
different token counts N at the same nominal budget, so they have different
cloud compute time -- see paper_data.T_CLOUD_BY_METHOD), not a single shared
per-budget value.

Usage:
    python plot_vqav2_figures.py             # both figures, S_fixed=1.0 Mbps
    python plot_vqav2_figures.py --s 2.5     # Figure 2 at a different bandwidth
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_data import BUDGETS, DATA, METHODS, COLORS, STYLES, WIDTHS, LABELS, N_BY_METHOD, T_CLOUD_BY_METHOD, T_EDGE


def to_latency(method: str, idx: int, s_mbps: float) -> float:
    budget = BUDGETS[idx]
    t_cloud = T_CLOUD_BY_METHOD[method][idx]
    t_tx = budget / (s_mbps * 1000)  # ms
    return T_EDGE + t_cloud + t_tx


def plot_figure1(out_path="figure1_per_benchmark.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (bench_name, methods) in zip(axes, DATA.items()):
        for method_name in METHODS:
            scores = methods[method_name]
            ax.plot(BUDGETS, scores,
                    label=LABELS[method_name],
                    color=COLORS[method_name],
                    linestyle=STYLES[method_name],
                    linewidth=WIDTHS[method_name],
                    marker='o', markersize=4)
        ax.set_xscale('log')
        ax.set_xlabel('Transmission Budget (bits, log scale)')
        ax.set_ylabel(f'{bench_name} Score')
        ax.set_title(bench_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


def plot_figure2(s_mbps: float, out_path="figure2_per_benchmark.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (bench_name, methods) in zip(axes, DATA.items()):
        for method_name in METHODS:
            scores = methods[method_name]
            latencies = [to_latency(method_name, i, s_mbps) for i in range(len(BUDGETS))]
            ax.plot(latencies, scores,
                    label=LABELS[method_name],
                    color=COLORS[method_name],
                    linestyle=STYLES[method_name],
                    linewidth=WIDTHS[method_name],
                    marker='o', markersize=4)
        ax.set_xlabel(f'T_total (ms) at S={s_mbps}Mbps')
        ax.set_ylabel(f'{bench_name} Score')
        ax.set_title(bench_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


def print_win_counts():
    summary = {}
    for bench_name, methods in DATA.items():
        wins = 0
        for i in range(len(BUDGETS)):
            scores_at_i = {m: v[i] for m, v in methods.items()}
            best_method = max(scores_at_i, key=scores_at_i.get)
            if best_method == "VQAv2-opt" or scores_at_i["VQAv2-opt"] == scores_at_i[best_method]:
                wins += 1
        summary[bench_name] = f"{wins}/{len(BUDGETS)}"
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=float, default=1.0, help="Uplink bandwidth in Mbps for Figure 2 (default: 1.0)")
    args = parser.parse_args()

    plot_figure1()
    plot_figure2(args.s)
    print_win_counts()
