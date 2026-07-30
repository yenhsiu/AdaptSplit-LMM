"""
Figure 3 only: Bandwidth vs. End-to-End Latency.
Reuses plot_result.py's existing load_data / compute_all / plot_bandwidth_vs_latency
-- no accuracy data needed, just the latency CSVs already in this directory.

Usage:
    python plot_figure3_only.py [--t-budget 600]
"""
import argparse
from plot_result import load_data, compute_all, plot_bandwidth_vs_latency, setup_paper_style, T_BUDGET

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-budget", type=float, default=T_BUDGET,
                        help=f"T_budget (ms) horizontal reference line (default: {T_BUDGET})")
    parser.add_argument("--save-path", default="figure3_bandwidth_latency.png")
    args = parser.parse_args()

    setup_paper_style()
    T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud, T_ori_jetson, T_ori_a100 = load_data()
    results = compute_all(T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud, T_ori_jetson, T_ori_a100)
    plot_bandwidth_vs_latency(results, T_budget=args.t_budget, save_path=args.save_path)
