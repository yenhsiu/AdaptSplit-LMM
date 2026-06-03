"""
Latency Analysis & Visualization
4 比較手法：
    1. Edge Inference
    2. Cloud Inference
    3. Split + PruMerge base + TurboQuant
    4. Split + PruMerge plus + TurboQuant
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons
import os

# ============================================================
# 設定
# ============================================================

FILE_EDGE       = "results_edge_Jetson_AGX_Orin.csv"
FILE_CLOUD      = "results_cloud.csv"
FILE_ORI_JETSON = "results_latency_ori_Jetson_AGX_Orin.csv"
FILE_ORI_A100   = "results_latency_ori_A100.csv"

IMAGE_SIZE_BITS = 336 * 336 * 3 * 8
TOKEN_DIM       = 1024
S_LIST = [0.5, 1, 2, 5, 10, 20]
B_LIST = [1, 2, 4]
N_BASE = 32
N_PLUS = 145
N_ORI  = 576

# ── 全域 latency budget（所有圖共用同一設定）──────────────────
T_BUDGET = 600   # ms

COLORS = {
    "edge":           "#EAB631",
    "cloud":          "#C0392B",
    "split_base":     "#14B582",
    "split_plus":     "#24A32C",
    "original_quant": "#6A3C98",
    "original_fp16":  "#1B56B5DE",
}

ORI_MARKERS = {4: 'o', 2: '^', 1: 'x'}


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4)


def setup_paper_style():
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.size':          11,
        'axes.titlesize':     13,
        'axes.labelsize':     12,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'legend.fontsize':    9,
        'legend.framealpha':  0.9,
        'legend.edgecolor':   '#cccccc',
        'axes.linewidth':     0.8,
        'grid.linewidth':     0.5,
        'grid.alpha':         0.35,
        'lines.linewidth':    1.8,
        'figure.dpi':         150,
        'savefig.dpi':        200,
        'savefig.bbox':       'tight',
    })

# ============================================================
# Accuracy データ読み込み（CSV から）
# ============================================================

def load_acc_data(csv_path="accuracy_results.csv"):
    df = pd.read_csv(csv_path)
    acc = {}
    for _, row in df.iterrows():
        dataset = row['dataset']
        category = row['category']
        bits = str(row['bits']).strip()
        score = float(row['score'])
        if dataset not in acc:
            acc[dataset] = {}
        if bits == '-':
            key = category
        else:
            key = f"{category}_B{bits}"
        acc[dataset][key] = score
    return acc

# ============================================================
# データ読み込み
# ============================================================

def load_data():
    # Edge コンポーネント
    df_edge = pd.read_csv(FILE_EDGE)

    T_edge = df_edge[(df_edge['component'] == 'T_edge')]['mean_ms'].values[0]
    T_pm_base = df_edge[(df_edge['component'] == 'T_prumerge_base')]['mean_ms'].values[0]
    T_pm_plus = df_edge[(df_edge['component'] == 'T_prumerge_plus')]['mean_ms'].values[0]

    # T_quant: {(N, B): mean_ms}
    df_quant = df_edge[df_edge['component'] == 'T_quant'].copy()
    T_quant = {
        (int(row['N']), int(row['B'])): row['mean_ms']
        for _, row in df_quant.iterrows()
    }

    # T_cloud: {N: mean_ms}
    df_cloud = pd.read_csv(FILE_CLOUD)
    T_cloud = {int(row['N']): row['mean_ms'] for _, row in df_cloud.iterrows()}

    # T_latency_ori
    T_ori_jetson = pd.read_csv(FILE_ORI_JETSON)['mean_ms'].values[0]
    T_ori_a100   = pd.read_csv(FILE_ORI_A100)['mean_ms'].values[0]

    return T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud, T_ori_jetson, T_ori_a100


# ============================================================
# T_total 計算
# ============================================================

def T_tx_image(S):
    return IMAGE_SIZE_BITS / (S * 1e6) * 1000

def T_tx_token(N, B, S):
    return N * TOKEN_DIM * B / (S * 1e6) * 1000

def compute_all(T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud, T_ori_jetson, T_ori_a100):
    results = {}

    # Edge Inference（S に依存しない）
    results['edge'] = {'label': 'Edge Inference', 'data': []}
    for S in S_LIST:
        results['edge']['data'].append({'S': S, 'T_total': T_ori_jetson, 'breakdown': {
            'T_compute': T_ori_jetson, 'T_tx': 0
        }})

    # Cloud Inference
    results['cloud'] = {'label': 'Cloud Inference', 'data': []}
    for S in S_LIST:
        tx = T_tx_image(S)
        total = tx + T_ori_a100
        results['cloud']['data'].append({'S': S, 'T_total': total, 'breakdown': {
            'T_tx': tx, 'T_compute': T_ori_a100
        }})

    # Split + PruMerge base + TurboQuant（B=1,2,4）
    results['split_base'] = {'label': 'Split + PruMerge base + TurboQuant', 'data': []}
    for S in S_LIST:
        for B in B_LIST:
            quant_t = T_quant.get((N_BASE, B), T_quant.get((32, B), 2.0))
            cloud_t = T_cloud.get(N_BASE, T_cloud.get(32, 24.0))
            tx = T_tx_token(N_BASE, B, S)
            total = T_edge + T_pm_base + quant_t + tx + cloud_t
            results['split_base']['data'].append({
                'S': S, 'B': B, 'T_total': total,
                'breakdown': {
                    'T_edge': T_edge,
                    'T_prumerge': T_pm_base,
                    'T_quant': quant_t,
                    'T_tx': tx,
                    'T_cloud': cloud_t
                }
            })

    # Split + PruMerge plus + TurboQuant（B=1,2,4）
    results['split_plus'] = {'label': 'Split + PruMerge plus + TurboQuant', 'data': []}
    for S in S_LIST:
        for B in B_LIST:
            quant_t = T_quant.get((N_PLUS, B), 5.0)
            cloud_t = T_cloud.get(N_PLUS, 28.0)
            tx = T_tx_token(N_PLUS, B, S)
            total = T_edge + T_pm_plus + quant_t + tx + cloud_t
            results['split_plus']['data'].append({
                'S': S, 'B': B, 'T_total': total,
                'breakdown': {
                    'T_edge': T_edge,
                    'T_prumerge': T_pm_plus,
                    'T_quant': quant_t,
                    'T_tx': tx,
                    'T_cloud': cloud_t
                }
            })

    # Original + TurboQuant（B=1,2,4, N=576, no prumerge）
    # T_quant for N=576 estimated by linear scaling from N_PLUS measurements
    results['original_quant'] = {'label': 'Original + TurboQuant', 'data': []}
    for S in S_LIST:
        for B in B_LIST:
            quant_t = T_quant.get(
                (N_ORI, B),
                T_quant.get((N_PLUS, B), 5.0) * N_ORI / N_PLUS
            )
            tx = T_tx_token(N_ORI, B, S)
            total = T_edge + quant_t + tx + T_ori_a100
            results['original_quant']['data'].append({
                'S': S, 'B': B, 'T_total': total,
                'breakdown': {
                    'T_edge': T_edge,
                    'T_prumerge': 0,
                    'T_quant': quant_t,
                    'T_tx': tx,
                    'T_cloud': T_ori_a100
                }
            })

    # Original split, no PruMerge, no TurboQuant（N=576, float16 = B=16）
    results['original_fp16'] = {'label': 'Original Split (float16)', 'data': []}
    for S in S_LIST:
        tx = T_tx_token(N_ORI, 16, S)
        cloud_t = T_cloud.get(N_ORI, T_ori_a100)
        total = T_edge + tx + cloud_t
        results['original_fp16']['data'].append({
            'S': S, 'T_total': total,
            'breakdown': {
                'T_edge': T_edge,
                'T_prumerge': 0,
                'T_quant': 0,
                'T_tx': tx,
                'T_cloud': cloud_t
            }
        })

    return results


# ============================================================
# Graph 1：帯域幅 vs T_total
# ============================================================

def plot_bandwidth_vs_latency(results, T_budget=T_BUDGET, save_path="graph1_bandwidth_latency.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Edge Inference（水平線）
    T_edge_val = results['edge']['data'][0]['T_total']
    ax.axhline(y=T_edge_val, color=COLORS['edge'], linewidth=2,
               linestyle='--', label=f"Edge Inference ({T_edge_val:.0f}ms)")

    # Cloud Inference
    s_vals = S_LIST
    cloud_totals = [d['T_total'] for d in results['cloud']['data']]
    ax.plot(s_vals, cloud_totals, color=COLORS['cloud'], linewidth=2,
            marker='o', markersize=5, label="Cloud Inference")

    # Proposed base（dynamic B per S）
    base_ser = get_proposed_series(results, 'split_base', N_BASE, T_budget)
    base_totals = [d['T_total'] for d in base_ser]
    ax.plot(s_vals, base_totals, color=COLORS['split_base'], linewidth=2,
            marker='s', markersize=5, label=f"Proposed base (dynamic B, N={N_BASE})")
    for d in base_ser:
        i = S_LIST.index(d['S'])
        ax.annotate(f"B={d['B']}", (d['S'], d['T_total']),
                    textcoords="offset points", xytext=(3, 4), fontsize=6,
                    color=COLORS['split_base'])

    # Proposed plus（dynamic B per S）
    plus_ser = get_proposed_series(results, 'split_plus', N_PLUS, T_budget)
    plus_totals = [d['T_total'] for d in plus_ser]
    ax.plot(s_vals, plus_totals, color=COLORS['split_plus'], linewidth=2,
            marker='^', markersize=5, label=f"Proposed plus (dynamic B, N={N_PLUS})")
    for d in plus_ser:
        ax.annotate(f"B={d['B']}", (d['S'], d['T_total']),
                    textcoords="offset points", xytext=(3, 4), fontsize=6,
                    color=COLORS['split_plus'])

    # Original + TurboQuant（B=1,2,4）
    line_styles = {4: ('-', 2.0, 5), 2: ('-.', 1.5, 4), 1: ('--', 1.5, 4)}
    for B, (ls, lw, ms) in line_styles.items():
        vals = [d['T_total'] for d in results['original_quant']['data'] if d['B'] == B]
        ax.plot(s_vals, vals, color=COLORS['original_quant'], linewidth=lw,
                marker=ORI_MARKERS[B], markersize=ms, linestyle=ls,
                label=f"Original+Quant B={B} (N={N_ORI})")

    ax.axhline(y=T_budget, color='black', linewidth=1.5, linestyle='-',
               label=f"T_budget = {T_budget:.0f} ms")

    ax.set_xlabel("Network Bandwidth (Mbps)")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title("Bandwidth vs. End-to-End Latency")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(S_LIST)
    ax.set_xticklabels([str(s) for s in S_LIST])
    ax.grid(True, which='major')
    ax.legend(loc='upper right', ncol=1)
    despine(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 2：T_total 内訳（Breakdown，S=5Mbps，B=4bit）
# ============================================================

def plot_breakdown(results, S_target=5, B_target=4, T_budget=T_BUDGET,
                   save_path="graph2_breakdown.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    components = ['T_tx', 'T_edge', 'T_prumerge', 'T_quant', 'T_cloud']
    comp_colors = {
        'T_tx':      '#BA7517',
        'T_edge':    '#378ADD',
        'T_prumerge':'#1D9E75',
        'T_quant':   '#D85A30',
        'T_cloud':   '#888780',
    }

    # 各手法のデータを取得
    method_labels = []
    method_data = []

    # Edge Inference
    T_edge_full = results['edge']['data'][0]['T_total']
    method_labels.append("Edge\nInference")
    method_data.append({
        'T_tx': 0, 'T_edge': T_edge_full,
        'T_prumerge': 0, 'T_quant': 0, 'T_cloud': 0
    })

    # Cloud
    cloud_d = next(d for d in results['cloud']['data'] if d['S'] == S_target)
    method_labels.append("Cloud\nInference")
    method_data.append({
        'T_tx': cloud_d['breakdown']['T_tx'],
        'T_edge': cloud_d['breakdown']['T_compute'],
        'T_prumerge': 0, 'T_quant': 0, 'T_cloud': 0
    })

    # Ori+Quant B=4
    ori_d4 = next(d for d in results['original_quant']['data']
                  if d['S'] == S_target and d['B'] == B_target)
    method_labels.append(f"Ori+Quant\nB={B_target}")
    method_data.append(ori_d4['breakdown'])

    # Ori+Quant B=2
    ori_d2 = next(d for d in results['original_quant']['data']
                  if d['S'] == S_target and d['B'] == 2)
    method_labels.append("Ori+Quant\nB=2")
    method_data.append(ori_d2['breakdown'])

    # Ori+Quant B=1
    ori_d1 = next(d for d in results['original_quant']['data']
                  if d['S'] == S_target and d['B'] == 1)
    method_labels.append("Ori+Quant\nB=1")
    method_data.append(ori_d1['breakdown'])

    # Proposed base（dynamic B at S_target）
    B_base = dynamic_b(N_BASE, S_target, T_budget)
    base_d = next(d for d in results['split_base']['data']
                  if d['S'] == S_target and d['B'] == B_base)
    method_labels.append(f"Proposed\nbase B={B_base}")
    method_data.append(base_d['breakdown'])

    # Proposed plus（dynamic B at S_target）
    B_plus = dynamic_b(N_PLUS, S_target, T_budget)
    plus_d = next(d for d in results['split_plus']['data']
                  if d['S'] == S_target and d['B'] == B_plus)
    method_labels.append(f"Proposed\nplus B={B_plus}")
    method_data.append(plus_d['breakdown'])

    n_bars = len(method_labels)
    fig_w = max(9, n_bars * 1.1)
    fig.set_size_inches(fig_w, 5)

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)

    comp_labels = {
        'T_tx':      'Transmission',
        'T_edge':    'Edge compute',
        'T_prumerge':'PruMerge',
        'T_quant':   'Quantization',
        'T_cloud':   'Cloud compute',
    }
    for comp in components:
        vals = [d.get(comp, 0) for d in method_data]
        ax.bar(x, vals, bottom=bottoms, label=comp_labels[comp],
               color=comp_colors[comp], alpha=0.88, width=0.55,
               edgecolor='white', linewidth=0.4)
        for i, v in enumerate(vals):
            if v > 15:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}",
                        ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')
        bottoms += np.array(vals)

    for i, total in enumerate(bottoms):
        ax.text(i, total * 1.03, f"{total:.0f} ms",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale('log')
    ax.set_title(f"Latency Breakdown\n(S = {S_target} Mbps)")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', which='major')
    despine(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 3：N vs T_total（token 数の影響）
# ============================================================

def plot_n_vs_latency(T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud,
                      S_target=1, B_target=None,
                      save_path="graph3_n_latency.png"):
    fig, ax = plt.subplots(figsize=(9, 5))

    N_vals = sorted(T_cloud.keys())
    b_list = B_LIST if B_target is None else [B_target]
    line_styles = {4: '-',  2: '--', 1: ':'}
    line_widths  = {4: 2.0, 2: 1.8, 1: 1.8}
    markers      = {4: 'o', 2: 'x',  1: '^'}
    marker_every = max(1, len(N_vals) // 8)

    for B in b_list:
        base_totals = []
        for N in N_vals:
            q = T_quant.get((N, B), 2.0)
            c = T_cloud.get(N, 24.0)
            tx = T_tx_token(N, B, S_target)
            base_totals.append(T_edge + T_pm_base + q + tx + c)
        ax.plot(N_vals, base_totals, color=COLORS['split_base'],
                linewidth=line_widths[B], linestyle=line_styles[B],
                marker=markers[B], markersize=5, markevery=marker_every,
                label=f"Split + PM base (B={B})")

    for B in b_list:
        plus_totals = []
        for N in N_vals:
            q = T_quant.get((N, B), 5.0)
            c = T_cloud.get(N, 28.0)
            tx = T_tx_token(N, B, S_target)
            plus_totals.append(T_edge + T_pm_plus + q + tx + c)
        ax.plot(N_vals, plus_totals, color=COLORS['split_plus'],
                linewidth=line_widths[B], linestyle=line_styles[B],
                marker=markers[B], markersize=5, markevery=marker_every,
                label=f"Split + PM plus (B={B})")

    # Use log scale for y-axis
    # ax.set_yscale('log')

    # N_BASE と N_PLUS の位置に縦線
    ax.axvline(x=N_BASE, color=COLORS['split_base'], linestyle=':', alpha=0.7,
               label=f"N_BASE (AVG={N_BASE})")
    ax.axvline(x=N_PLUS, color=COLORS['split_plus'], linestyle=':', alpha=0.7,
               label=f"N_PLUS (AVG={N_PLUS})")

    ax.set_xlabel("Number of Visual Tokens N", fontsize=12)
    ax.set_ylabel("End-to-End Latency (ms)", fontsize=12)
    ax.set_title(f"Token Count vs Latency (S={S_target}Mbps)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 4：Acc vs Latency
# ============================================================

def _plot_proposed_acc_series(ax, results, method, N_tokens, acc, prefix, color, marker, T_budget):
    """Plot dynamic-B proposed curve on acc vs latency graph."""
    ser = get_proposed_series(results, method, N_tokens, T_budget)
    pts_T = [d['T_total'] for d in ser]
    pts_acc = [acc.get(f"{method}_B{d['B']}") for d in ser]
    valid = [(t, a, d['B']) for t, a, d in zip(pts_T, pts_acc, ser)
             if a is not None and t <= T_budget]
    if not valid:
        return
    Ts, accs, Bs = zip(*valid)
    ax.scatter(Ts, accs, color=color, marker=marker, s=60, zorder=5, label=f"{prefix} (dynamic B)")
    ax.plot(Ts, accs, color=color, linewidth=1.5, alpha=0.6)
    for T, a, B in zip(Ts, accs, Bs):
        ax.annotate(f"B={B}", (T, a),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=6, color=color)


def plot_acc_vs_latency(results, benchmark="MME", acc_data=None, T_budget=T_BUDGET,
                        save_path=None):
    if save_path is None:
        save_path = f"graph4_acc_latency_{benchmark}.png"

    acc = acc_data or {}
    ylabel = "Accuracy (%)" if benchmark in ("POPE", "SQA") else "MME Score"

    fig, ax = plt.subplots(figsize=(8, 5))

    # Edge Inference
    T_edge_val = results['edge']['data'][0]['T_total']
    if 'edge' in acc and T_edge_val <= T_budget:
        ax.scatter([T_edge_val], [acc['edge']],
                   color=COLORS['edge'], marker='P', s=120, zorder=5,
                   label="Edge Inference")
        ax.annotate("", (T_edge_val, acc['edge']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Cloud Inference
    cloud_totals = [d['T_total'] for d in results['cloud']['data'] if d['T_total'] <= T_budget]
    cloud_s = [d['S'] for d in results['cloud']['data'] if d['T_total'] <= T_budget]
    if 'cloud' in acc and cloud_totals:
        ax.scatter(cloud_totals, [acc['cloud']] * len(cloud_totals),
                   color=COLORS['cloud'], marker='D', s=60, zorder=5,
                   label="Cloud Inference")
        ax.plot(cloud_totals, [acc['cloud']] * len(cloud_totals),
                color=COLORS['cloud'], linewidth=1, alpha=0.5)
        for i, (S, T) in enumerate(zip(cloud_s, cloud_totals)):
            if i in [0, len(cloud_totals)-1]:
                ax.annotate(f"S={S}", (T, acc['cloud']),
                            textcoords="offset points", xytext=(3, -12),
                            fontsize=7, color=COLORS['cloud'])

    # Proposed base（dynamic B）
    _plot_proposed_acc_series(ax, results, 'split_base', N_BASE, acc,
                               "Proposed base", COLORS['split_base'], 's', T_budget)

    # Proposed plus（dynamic B）
    _plot_proposed_acc_series(ax, results, 'split_plus', N_PLUS, acc,
                               "Proposed plus", COLORS['split_plus'], '^', T_budget)

    # Original + TurboQuant（B=1,2,4）
    if 'original_quant' in results:
        for B in B_LIST:
            key = f"original_quant_B{B}"
            if key not in acc:
                continue
            acc_val = acc[key]
            totals = [d['T_total'] for d in results['original_quant']['data']
                      if d['B'] == B and d['T_total'] <= T_budget]
            if not totals:
                continue
            mk = ORI_MARKERS[B]
            ax.scatter(totals, [acc_val] * len(totals), color=COLORS['original_quant'],
                       marker=mk, s=60, alpha=0.85, zorder=5,
                       label=f"Ori+Quant B={B}")
            ax.plot(totals, [acc_val] * len(totals),
                    color=COLORS['original_quant'], linewidth=1, alpha=0.35, linestyle='--')
            mid = len(totals) // 2
            ax.annotate(f"B={B}", (totals[mid], acc_val),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7, color=COLORS['original_quant'])

    s_str = ", ".join(str(s) for s in S_LIST)
    ax.set_xlabel("End-to-End Latency (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Accuracy vs. Latency  ({benchmark})\nS = [{s_str}] Mbps")
    ax.legend(loc='lower right', ncol=1)
    ax.grid(True, which='major')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    despine(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 4 (interactive)：ベンチマーク切り替え可能
# ============================================================

def _draw_acc_vs_latency(ax, results, benchmark, acc_data, T_budget=T_BUDGET):
    ax.clear()
    acc = acc_data.get(benchmark, {})
    ylabel = "Accuracy (%)" if benchmark == "POPE" else "MME Score"

    # Edge Inference
    T_edge_val = results['edge']['data'][0]['T_total']
    if 'edge' in acc and T_edge_val <= T_budget:
        ax.scatter([T_edge_val], [acc['edge']],
                   color=COLORS['edge'], marker='o', s=120, zorder=5,
                   label="Edge Inference")
        ax.annotate("", (T_edge_val, acc['edge']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Cloud Inference
    cloud_totals = [d['T_total'] for d in results['cloud']['data'] if d['T_total'] <= T_budget]
    cloud_s = [d['S'] for d in results['cloud']['data'] if d['T_total'] <= T_budget]
    if 'cloud' in acc and cloud_totals:
        ax.scatter(cloud_totals, [acc['cloud']] * len(cloud_totals),
                   color=COLORS['cloud'], marker='D', s=60, zorder=5,
                   label="Cloud Inference")
        ax.plot(cloud_totals, [acc['cloud']] * len(cloud_totals),
                color=COLORS['cloud'], linewidth=1, alpha=0.5)
        for i, (S, T) in enumerate(zip(cloud_s, cloud_totals)):
            if i in [0, len(cloud_totals) - 1]:
                ax.annotate(f"S={S}", (T, acc['cloud']),
                            textcoords="offset points", xytext=(3, -12),
                            fontsize=7, color=COLORS['cloud'])

    # Proposed base（dynamic B）
    _plot_proposed_acc_series(ax, results, 'split_base', N_BASE, acc,
                               "Proposed base", COLORS['split_base'], 's', T_budget)

    # Proposed plus（dynamic B）
    _plot_proposed_acc_series(ax, results, 'split_plus', N_PLUS, acc,
                               "Proposed plus", COLORS['split_plus'], '^', T_budget)

    # Original + TurboQuant（B=1,2,4）
    if 'original_quant' in results:
        for B in B_LIST:
            key = f"original_quant_B{B}"
            if key not in acc:
                continue
            acc_val = acc[key]
            totals = [d['T_total'] for d in results['original_quant']['data']
                      if d['B'] == B and d['T_total'] <= T_budget]
            if not totals:
                continue
            mk = ORI_MARKERS[B]
            ax.scatter(totals, [acc_val] * len(totals), color=COLORS['original_quant'],
                       marker=mk, s=60, alpha=0.85, zorder=5,
                       label=f"Ori+Quant B={B}")
            ax.plot(totals, [acc_val] * len(totals),
                    color=COLORS['original_quant'], linewidth=1, alpha=0.35, linestyle='--')
            mid = len(totals) // 2
            ax.annotate(f"B={B}", (totals[mid], acc_val),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7, color=COLORS['original_quant'])

    s_str = ", ".join(str(s) for s in S_LIST)
    ax.set_xlabel("End-to-End Latency (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Accuracy vs. Latency  ({benchmark})\nS = [{s_str}] Mbps")
    ax.legend(loc='lower right', ncol=1)
    ax.grid(True, which='major')
    ax.set_xscale('log')
    despine(ax)


def plot_acc_vs_latency_interactive(results, acc_data):
    benchmarks = list(acc_data.keys())

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.18, 0.1, 0.78, 0.82])
    rax = fig.add_axes([0.01, 0.4, 0.14, 0.2])
    radio = RadioButtons(rax, benchmarks, activecolor=COLORS['split_base'])

    _draw_acc_vs_latency(ax, results, benchmarks[0], acc_data)

    def on_click(label):
        _draw_acc_vs_latency(ax, results, label, acc_data)
        fig.canvas.draw_idle()

    radio.on_clicked(on_click)
    plt.suptitle("Accuracy vs Latency  (select benchmark →)", fontsize=11, x=0.55)
    plt.show()


# ============================================================
# Dynamic B selection
# ============================================================

def dynamic_b(N, S, budget, b_list=None):
    if b_list is None:
        b_list = [4, 2, 1]
    for B in sorted(b_list, reverse=True):
        if T_tx_token(N, B, S) <= budget:
            return B
    return min(b_list)


def get_acc_by_b(acc_data, benchmark, split_type="split_plus"):
    d = acc_data.get(benchmark, {})
    return {B: d[f"{split_type}_B{B}"]
            for B in [1, 2, 4]
            if f"{split_type}_B{B}" in d}


def get_proposed_series(results, method, N_tokens, T_budget):
    """For each S, pick the highest B where T_total <= T_budget."""
    out = []
    for S in S_LIST:
        candidates = sorted(
            [x for x in results[method]['data'] if x['S'] == S and x['T_total'] <= T_budget],
            key=lambda x: x['B'], reverse=True
        )
        if not candidates:
            continue
        best = candidates[0]
        out.append({'S': S, 'B': best['B'], 'T_total': best['T_total'],
                    'breakdown': best['breakdown']})
    return out


def plot_dynamic_n_vs_ttx(T_budget, acc_by_b, benchmark="POPE",
                           S_target=5,
                           save_path="dynamic_graph1_N_vs_Ttx.png"):
    N_vals = np.arange(10, 580, 5)
    b_list = sorted(acc_by_b.keys())

    ttx_b4   = [T_tx_token(N, 4, S_target) for N in N_vals]
    ttx_b2   = [T_tx_token(N, 2, S_target) for N in N_vals]
    ttx_b1   = [T_tx_token(N, 1, S_target) for N in N_vals]
    ttx_dyn  = [T_tx_token(N, dynamic_b(N, S_target, T_budget, b_list), S_target) for N in N_vals]
    b_chosen = [dynamic_b(N, S_target, T_budget, b_list) for N in N_vals]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(N_vals, ttx_b4,  color='#A32D2D', linewidth=2, label="Fixed B=4")
    ax1.plot(N_vals, ttx_b2,  color='#BA7517', linewidth=1.5, label="Fixed B=2", linestyle='-.')
    ax1.plot(N_vals, ttx_b1,  color='#888780', linewidth=2, label="Fixed B=1", linestyle='--')
    ax1.plot(N_vals, ttx_dyn, color='#1D9E75', linewidth=2.5, label="Dynamic b*")
    ax1.axhline(y=T_budget, color='black', linewidth=1.5, linestyle=':',
                label=f"T_budget = {T_budget:.0f}ms (T_edge)")
    ax1.axvline(x=N_BASE, color='#378ADD', linewidth=1, linestyle=':', alpha=0.7,
                label=f"PM base avg (N={N_BASE})")
    ax1.axvline(x=N_PLUS, color='#BA7517', linewidth=1, linestyle=':', alpha=0.7,
                label=f"PM plus avg (N={N_PLUS})")
    ax1.axvspan(10, 580, alpha=0.05, color='#378ADD')
    ax1.axvspan(30, 580, alpha=0.05, color='#BA7517')
    ax1.set_ylabel("T_tx (ms)", fontsize=12)
    ax1.set_title(f"Token Count N vs T_tx  (S={S_target} Mbps)\n"
                  f"Fixed B=4 explodes at large N; Dynamic b* stays within budget",
                  fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, which='major')
    ax1.set_xlim(10, 580)
    despine(ax1)

    ax2.step(N_vals, b_chosen, color='#1D9E75', linewidth=2, where='post')
    ax2.set_yticks(sorted(b_list))
    ax2.set_yticklabels([f"B={b}" for b in sorted(b_list)])
    ax2.set_xlabel("Number of Visual Tokens $N$")
    ax2.set_ylabel("Selected $b^*$")
    ax2.grid(True, which='major')
    ax2.set_xlim(10, 580)
    despine(ax2)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def plot_dynamic_scenarios(T_budget, acc_by_b, benchmark="POPE",
                            save_path="dynamic_graph2_scenarios.png"):
    scenarios = [
        {"label": "Scenario 1\nN=30\nS=5Mbps",   "N": 30,  "S": 5.0},
        {"label": "Scenario 2\nN=145\nS=5Mbps",  "N": 145, "S": 5.0},
        {"label": "Scenario 3\nN=350\nS=5Mbps",  "N": 350, "S": 5.0},
        {"label": "Scenario 4\nN=145\nS=1Mbps",  "N": 145, "S": 1.0},
        {"label": "Scenario 5\nN=350\nS=1Mbps",  "N": 350, "S": 1.0},
        {"label": "Scenario 6\nN=100\nS=0.5Mbps","N": 100, "S": 0.5},
    ]
    b_list = sorted(acc_by_b.keys())

    ttx_b4   = [T_tx_token(s["N"], 4, s["S"]) for s in scenarios]
    ttx_b2   = [T_tx_token(s["N"], 2, s["S"]) for s in scenarios]
    ttx_b1   = [T_tx_token(s["N"], 1, s["S"]) for s in scenarios]
    ttx_dyn  = [T_tx_token(s["N"], dynamic_b(s["N"], s["S"], T_budget, b_list), s["S"]) for s in scenarios]
    b_chosen = [dynamic_b(s["N"], s["S"], T_budget, b_list) for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - 1.5*width, ttx_b4,  width, label="Fixed B=4", color='#A32D2D', alpha=0.85)
    ax.bar(x - 0.5*width, ttx_b2,  width, label="Fixed B=2", color='#BA7517', alpha=0.85)
    ax.bar(x + 0.5*width, ttx_b1,  width, label="Fixed B=1", color='#888780', alpha=0.85)
    ax.bar(x + 1.5*width, ttx_dyn, width, label="Dynamic b*", color='#1D9E75', alpha=0.85)
    ax.axhline(y=T_budget, color='black', linewidth=1.5, linestyle=':',
               label=f"T_budget = {T_budget:.0f}ms")

    for i, (b4, dyn, b) in enumerate(zip(ttx_b4, ttx_dyn, b_chosen)):
        ax.text(i - 1.5*width, b4 + 10, f"{b4:.0f}ms",
                ha='center', va='bottom', fontsize=7.5, color='#A32D2D')
        ax.text(i + 1.5*width, dyn + 10, f"{dyn:.0f}ms\n(B={b})",
                ha='center', va='bottom', fontsize=7.5, color='#1D9E75')

    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in scenarios], fontsize=9)
    ax.set_ylabel("$T_{tx}$ (ms)")
    ax.set_title("Transmission Latency: Fixed B vs. Dynamic $b^*$ Across Scenarios")
    ax.legend()
    ax.grid(True, axis='y', which='major')
    despine(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def plot_dynamic_b_heatmap(T_budget, T_edge, T_pm, T_quant, T_cloud,
                           method='split_base',
                           save_path="dynamic_graph3_b_heatmap.png"):
    """
    T_total based dynamic B heatmap。

    Parameters
    ----------
    T_edge  : float   edge compute latency (ms)
    T_pm    : float   PruMerge latency (ms)，split_base 傳 T_pm_base，split_plus 傳 T_pm_plus
    T_quant : dict    {(N, B): ms}
    T_cloud : dict    {N: ms}
    method  : str     'split_base' | 'split_plus'（用於 title 標示）
    """
    N_vals = np.arange(10, 580, 10)
    S_vals = [0.1, 0.5, 1, 2, 5, 10, 20, 50]

    # 建立插值用的已知 N 點（T_quant 和 T_cloud）
    known_N_cloud = sorted(T_cloud.keys())
    cloud_vals = np.array([T_cloud[n] for n in known_N_cloud], dtype=float)

    def interp_quant(N, B):
        known_N_q = sorted(set(k[0] for k in T_quant if k[1] == B))
        if not known_N_q:
            return 2.0
        q_vals = np.array([T_quant[(n, B)] for n in known_N_q], dtype=float)
        return float(np.interp(N, known_N_q, q_vals))

    def interp_cloud(N):
        return float(np.interp(N, known_N_cloud, cloud_vals))

    def best_b_total(N, S):
        for B in sorted(B_LIST, reverse=True):
            q = interp_quant(N, B)
            c = interp_cloud(N)
            tx = T_tx_token(N, B, S)
            if T_edge + T_pm + q + tx + c <= T_budget:
                return B
        return 0  # 全部超出 budget → 顯示為 0

    n_rows, n_cols = len(S_vals), len(N_vals)
    b_matrix = np.zeros((n_rows, n_cols))
    for i, S in enumerate(S_vals):
        for j, N in enumerate(N_vals):
            b_matrix[i, j] = best_b_total(N, S)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap('RdYlGn', 4)   # 0=超出(灰), 1=B=1(紅), 2=B=2(黃), 4=B=4(綠)
    im = ax.imshow(b_matrix, aspect='auto', cmap=cmap, vmin=-0.5, vmax=4.5)

    ax.set_xticks(np.arange(0, n_cols, max(1, n_cols // 10)))
    ax.set_xticklabels(N_vals[::max(1, n_cols // 10)], fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{s} Mbps" for s in S_vals], fontsize=9)
    ax.set_xlabel("Number of Visual Tokens $N$")
    ax.set_ylabel("Bandwidth $S$")
    ax.set_title(f"Dynamic $b^*$ Selection Map  [{method}, T_budget={T_budget:.0f}ms]\n"
                 f"Green=B=4, Yellow=B=2, Red=B=1, Dark=over budget")

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 4], fraction=0.03, pad=0.02)
    cbar.set_label("Selected $b^*$")
    cbar.ax.set_yticklabels(['Over budget', 'B=1', 'B=2', 'B=4'])

    # N_base / N_plus 標線（轉換為 pixel 座標）
    step = N_vals[1] - N_vals[0]
    n_base_px = (N_BASE - N_vals[0]) / step - 0.5
    n_plus_px = (N_PLUS - N_vals[0]) / step - 0.5
    ax.axvline(x=n_base_px, color='#2471A3', linewidth=1.5, linestyle='--', label=f'N_base={N_BASE}')
    ax.axvline(x=n_plus_px, color='#BA7517', linewidth=1.5, linestyle='--', label=f'N_plus={N_PLUS}')
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def plot_dynamic_timeseries(T_budget, acc_by_b, benchmark="POPE",
                             save_path="dynamic_graph4_timeseries.png"):
    np.random.seed(42)
    n_steps = 30
    b_list = sorted(acc_by_b.keys())

    N_series = np.clip(np.random.normal(145, 60, n_steps).astype(int), 30, 350)
    S_series = np.clip(np.random.normal(3, 2, n_steps), 0.3, 15)

    ttx_b4   = [T_tx_token(N, 4, S) for N, S in zip(N_series, S_series)]
    ttx_b2   = [T_tx_token(N, 2, S) for N, S in zip(N_series, S_series)]
    ttx_b1   = [T_tx_token(N, 1, S) for N, S in zip(N_series, S_series)]
    ttx_dyn  = [T_tx_token(N, dynamic_b(N, S, T_budget, b_list), S) for N, S in zip(N_series, S_series)]

    steps = np.arange(n_steps)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8),
                              gridspec_kw={'height_ratios': [1, 1, 2]})

    axes[0].plot(steps, N_series, color='#BA7517', linewidth=1.5)
    axes[0].axhline(y=N_BASE, color='#378ADD', linewidth=1, linestyle='--', alpha=0.7, label=f'base avg')
    axes[0].axhline(y=N_PLUS, color='#BA7517', linewidth=1, linestyle='--', alpha=0.7, label=f'plus avg')
    axes[0].set_ylabel("N (tokens)", fontsize=10)
    axes[0].set_title("Simulated N and S variation over time", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, n_steps - 1)

    axes[1].plot(steps, S_series, color='#888780', linewidth=1.5)
    axes[1].set_ylabel("S (Mbps)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, n_steps - 1)

    axes[2].plot(steps, ttx_b4,  color='#A32D2D', linewidth=2, label="Fixed B=4", alpha=0.9)
    axes[2].plot(steps, ttx_b2,  color='#BA7517', linewidth=1.5, label="Fixed B=2", alpha=0.7, linestyle='-.')
    axes[2].plot(steps, ttx_b1,  color='#888780', linewidth=1.5, label="Fixed B=1", alpha=0.7, linestyle='--')
    axes[2].plot(steps, ttx_dyn, color='#1D9E75', linewidth=2.5, label="Dynamic b*", alpha=0.9)
    axes[2].axhline(y=T_budget, color='black', linewidth=1.5, linestyle=':',
                    label=f"T_budget={T_budget:.0f}ms")
    for i, t in enumerate(ttx_b4):
        if t > T_budget:
            axes[2].axvspan(i - 0.4, i + 0.4, alpha=0.15, color='red')
    over_b4  = sum(1 for t in ttx_b4 if t > T_budget)
    over_dyn = sum(1 for t in ttx_dyn if t > T_budget)
    axes[2].text(0.98, 0.95,
                 f"Exceeded budget:\nFixed B=4: {over_b4}/{n_steps}\nDynamic b*: {over_dyn}/{n_steps}",
                 transform=axes[2].transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_ylabel("T_tx (ms)", fontsize=10)
    axes[2].set_xlabel("Time Step", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, which='major')
    axes[2].set_xlim(0, n_steps - 1)
    for a in axes:
        despine(a)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 5b：Acc vs Bandwidth（只顯示符合 latency budget 的點）
# ============================================================

def plot_acc_vs_bandwidth(results, acc_data, benchmark, target_budget,
                          save_path=None):
    """
    x 軸為頻寬（Mbps），y 軸為 accuracy，
    只畫出 T_total <= target_budget 的 (S, method/B) 組合。

    Parameters
    ----------
    results      : dict    compute_all() 的輸出。
    acc_data     : dict    load_acc_data() 的輸出（全 benchmark）。
    benchmark    : str     e.g. 'MME', 'POPE', 'SQA'。
    target_budget: float   latency 上限（ms），超過的點不畫。
    save_path    : str     None 則自動命名。
    """
    if save_path is None:
        save_path = f"graph_acc_vs_bw_{benchmark}_{int(target_budget)}ms.png"

    acc = acc_data.get(benchmark, {})
    if not acc:
        print(f"  WARNING: no acc data for benchmark '{benchmark}'")
        return

    ylabel = "Accuracy (%)" if benchmark in ("POPE", "SQA") else "MME Score"
    fig, ax = plt.subplots(figsize=(9, 5))

    # --- edge（不依賴頻寬）---
    T_edge_val = results['edge']['data'][0]['T_total']
    if T_edge_val <= target_budget and 'edge' in acc:
        ax.axhline(y=acc['edge'], color=COLORS['edge'], linewidth=2,
                   linestyle='--', label=f"Edge Inference ({T_edge_val:.0f} ms)")

    # --- cloud ---
    if 'cloud' in acc:
        xs, ys = [], []
        for d in results['cloud']['data']:
            if d['T_total'] <= target_budget:
                xs.append(d['S'])
                ys.append(acc['cloud'])
        if xs:
            ax.plot(xs, ys, color=COLORS['cloud'], linewidth=2,
                    marker='D', markersize=6, label="Cloud Inference")

    # --- split_base / split_plus（每個 S 選 dynamic B，再檢查 budget）---
    for method, N_tokens, mk in [
        ('split_base', N_BASE, 's'),
        ('split_plus', N_PLUS, '^'),
    ]:
        xs, ys, b_annots = [], [], []
        for d in results[method]['data']:
            S, B, T = d['S'], d['B'], d['T_total']
            if T > target_budget:
                continue
            key = f"{method}_B{B}"
            if key not in acc:
                continue
            # 同一個 S 只保留 B 最大的合法點
            existing = next((i for i, x in enumerate(xs) if x == S), None)
            if existing is not None:
                if B > b_annots[existing]:
                    xs[existing], ys[existing], b_annots[existing] = S, acc[key], B
            else:
                xs.append(S)
                ys.append(acc[key])
                b_annots.append(B)
        if xs:
            label = METHOD_NAMES.get(method, method)
            ax.plot(xs, ys, color=COLORS[method], linewidth=2,
                    marker=mk, markersize=6, label=label)
            for x, y, B in zip(xs, ys, b_annots):
                ax.annotate(f"B={B}", (x, y),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=COLORS[method])

    # --- original_quant（B=1,2,4 分別畫）---
    for B in B_LIST:
        key = f"original_quant_B{B}"
        if key not in acc:
            continue
        xs = [d['S'] for d in results['original_quant']['data']
              if d['B'] == B and d['T_total'] <= target_budget]
        if xs:
            ys = [acc[key]] * len(xs)
            ax.plot(xs, ys, color=COLORS['original_quant'], linewidth=1.5,
                    marker=ORI_MARKERS[B], markersize=6,
                    linestyle='--', alpha=0.85,
                    label=f"Original + TurboQuant B={B}")

    # --- original_fp16 ---
    if 'original_fp16' in acc:
        xs = [d['S'] for d in results['original_fp16']['data']
              if d['T_total'] <= target_budget]
        if xs:
            ys = [acc['original_fp16']] * len(xs)
            ax.plot(xs, ys, color=COLORS['original_fp16'], linewidth=1.5,
                    marker='x', markersize=6, label="Original Split (float16)")

    ax.set_xscale('log')
    ax.set_xticks(S_LIST)
    ax.set_xticklabels([str(s) for s in S_LIST])
    ax.set_xlabel("Network Bandwidth (Mbps)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Accuracy vs. Bandwidth  ({benchmark})\n"
                 f"Only methods with T_total ≤ {target_budget:.0f} ms shown")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, which='major', alpha=0.35)
    despine(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 5：指定手法在不同頻寬下的 Latency 變化
# ============================================================

METHOD_NAMES = {
    'edge':           'Edge Inference',
    'cloud':          'Cloud Inference',
    'split_base':     'Split + PruMerge base + TurboQuant',
    'split_plus':     'Split + PruMerge plus + TurboQuant',
    'original_quant': 'Original + TurboQuant',
    'original_fp16':  'Original Split (float16, no quant)',
}

# ============================================================
# Graph 6：單一手法在不同頻寬下的柱狀圖
# ============================================================

def plot_single_method_bar(results, method, T_budget=T_BUDGET,
                           save_path=None):
    """
    畫出單一手法在各頻寬下的柱狀圖，並以水平線標示 Edge Inference 作為基準。

    Parameters
    ----------
    results : dict
        compute_all() 的輸出。
    method : str
        手法 key：'cloud' | 'split_base' | 'split_plus' | 'original_quant'
    T_budget : float
        dynamic B 預算（ms），用於 split_base / split_plus。
    save_path : str or None
        None 則自動命名為 graph6_bar_{method}.png。
    """
    if save_path is None:
        save_path = f"graph6_bar_{method}.png"

    if method not in results:
        print(f"  ERROR: method '{method}' not found. Available: {list(results.keys())}")
        return

    color = COLORS[method]
    label_prefix = METHOD_NAMES.get(method, method)
    T_edge_val = results['edge']['data'][0]['T_total']

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(S_LIST))

    def _label_bar(bar, text, c='black'):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                text, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=c)

    if method == 'cloud':
        width = 0.5
        totals = [d['T_total'] for d in results['cloud']['data']]
        bars = ax.bar(x, totals, width, color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5, label=label_prefix)
        for bar, val in zip(bars, totals):
            _label_bar(bar, f"{val:.0f}", color)

    elif method in ('split_base', 'split_plus'):
        width = 0.5
        N_tokens = N_BASE if method == 'split_base' else N_PLUS
        ser = get_proposed_series(results, method, N_tokens, T_budget)
        totals = [d['T_total'] for d in ser]
        b_labels = [d['B'] for d in ser]
        bars = ax.bar(x, totals, width, color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5, label=label_prefix)
        for bar, val, B in zip(bars, totals, b_labels):
            _label_bar(bar, f"{val:.0f}\nB={B}", color)

    elif method == 'original_quant':
        sub_width = 0.6 / 3
        offsets = [-sub_width, 0, sub_width]
        b_colors = {4: '#7D3C98', 2: '#A569BD', 1: '#D2B4DE'}
        for B, offset in zip([4, 2, 1], offsets):
            vals = [d['T_total'] for d in results['original_quant']['data'] if d['B'] == B]
            bars = ax.bar(x + offset, vals, sub_width * 0.9,
                          color=b_colors[B], alpha=0.88, edgecolor='white',
                          linewidth=0.4, label=f"B={B}")
            for bar, val in zip(bars, vals):
                _label_bar(bar, f"{val:.0f}", b_colors[B])
        ax.legend(title="Bits", fontsize=9)

    elif method == 'original_fp16':
        width = 0.5
        totals = [d['T_total'] for d in results['original_fp16']['data']]
        bars = ax.bar(x, totals, width, color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5, label=label_prefix)
        for bar, val in zip(bars, totals):
            _label_bar(bar, f"{val:.0f}", color)

    # Edge 用水平線標示（不受頻寬影響，不需要單獨畫柱）
    ax.axhline(y=T_edge_val, color=COLORS['edge'], linewidth=2, linestyle='--',
               label=f"Edge Inference ({T_edge_val:.0f} ms)")
    ax.legend(fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} Mbps" for s in S_LIST])
    ax.set_xlabel("Network Bandwidth (Mbps)")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_yscale('log')
    ax.set_title(f"Latency vs. Bandwidth\n{label_prefix}")
    ax.grid(True, axis='y', which='major', alpha=0.4)
    despine(ax)


    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    setup_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="all",
                        help="Dataset to plot: MME | POPE | all (default: all)")
    parser.add_argument("--acc-file", default="accuracy_results.csv",
                        help="Path to accuracy CSV (default: accuracy_results.csv)")
    parser.add_argument("--dynamic-benchmark", default="POPE",
                        help="Benchmark for dynamic B graphs (default: POPE)")
    parser.add_argument("--dynamic-split", default="split_plus",
                        help="Split type for dynamic B acc lookup (default: split_plus)")
    parser.add_argument("--t-budget", type=float, default=T_BUDGET,
                        help=f"T_budget (ms) for dynamic B selection (default: {T_BUDGET}). "
                             "Use -1 to auto-set from T_ori_jetson.")
    args = parser.parse_args()

    print("Loading latency data...")
    T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud, T_ori_jetson, T_ori_a100 = load_data()

    print(f"  T_edge:        {T_edge:.1f} ms")
    print(f"  T_pm_base:     {T_pm_base:.1f} ms")
    print(f"  T_pm_plus:     {T_pm_plus:.1f} ms")
    print(f"  T_ori_jetson:  {T_ori_jetson:.1f} ms")
    print(f"  T_ori_a100:    {T_ori_a100:.1f} ms")
    print(f"  T_cloud(N={N_BASE}): {T_cloud.get(N_BASE, '?'):.1f} ms")
    print(f"  T_cloud(N={N_PLUS}): {T_cloud.get(N_PLUS, '?'):.1f} ms")

    print(f"\nLoading accuracy data from {args.acc_file}...")
    ACC_DATA = load_acc_data(args.acc_file)
    print(f"  Available benchmarks: {list(ACC_DATA.keys())}")

    T_budget = T_ori_jetson if args.t_budget < 0 else args.t_budget
    print(f"  T_budget:      {T_budget:.1f} ms")

    print("\nComputing T_total for all methods...")
    results = compute_all(T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud,
                          T_ori_jetson, T_ori_a100)

    print("\nGenerating graphs...")
    plot_bandwidth_vs_latency(results, T_budget=T_budget)
    for S_bd in S_LIST:
        plot_breakdown(results, S_target=S_bd, B_target=4, T_budget=T_budget,
                       save_path=f"graph2_breakdown_S{S_bd}Mbps.png")
    plot_n_vs_latency(T_edge, T_pm_base, T_pm_plus, T_quant, T_cloud,
                      S_target=1)

    if args.benchmark == "all":
        for bm in ACC_DATA:
            plot_acc_vs_latency(results, benchmark=bm, acc_data=ACC_DATA[bm],
                                T_budget=T_budget)
    else:
        if args.benchmark not in ACC_DATA:
            print(f"ERROR: '{args.benchmark}' not in {args.acc_file}, available: {list(ACC_DATA.keys())}")
            exit(1)
        plot_acc_vs_latency(results, benchmark=args.benchmark,
                            acc_data=ACC_DATA[args.benchmark], T_budget=T_budget)

    print("\nGenerating dynamic B selection graphs...")
    dyn_bm = args.dynamic_benchmark
    acc_by_b = get_acc_by_b(ACC_DATA, dyn_bm, args.dynamic_split)
    if acc_by_b:
        plot_dynamic_n_vs_ttx(T_budget, acc_by_b, benchmark=dyn_bm)
        plot_dynamic_scenarios(T_budget, acc_by_b, benchmark=dyn_bm)
        plot_dynamic_b_heatmap(T_budget, T_edge, T_pm_base, T_quant, T_cloud,
                               method='split_base')
        plot_dynamic_b_heatmap(T_budget, T_edge, T_pm_plus, T_quant, T_cloud,
                               method='split_plus',
                               save_path="dynamic_graph3_b_heatmap_plus.png")
        plot_dynamic_timeseries(T_budget, acc_by_b, benchmark=dyn_bm)
    else:
        print(f"  WARNING: no acc data for {dyn_bm}/{args.dynamic_split}, skipping dynamic graphs")

    # 只畫 cloud inference 的柱狀圖
    plot_single_method_bar(results, method='split_plus', T_budget=T_budget)
    plot_single_method_bar(results, method='split_base', T_budget=T_budget)
    plot_single_method_bar(results, method='original_fp16', T_budget=T_budget)
    plot_single_method_bar(results, method='original_quant', T_budget=T_budget)
    # 'split_base''split_plus','original_quant'	

    plot_acc_vs_bandwidth(results, ACC_DATA, benchmark='POPE', target_budget=T_BUDGET)
    plot_acc_vs_bandwidth(results, ACC_DATA, benchmark='MME', target_budget=T_BUDGET)
    plot_acc_vs_bandwidth(results, ACC_DATA, benchmark='SQA', target_budget=T_BUDGET)
    plot_acc_vs_bandwidth(results, ACC_DATA, benchmark='TextVQA', target_budget=T_BUDGET)
    plot_acc_vs_bandwidth(results, ACC_DATA, benchmark='VQAv2', target_budget=T_BUDGET)
    plot_acc_vs_bandwidth(results, ACC_DATA, benchmark='MMBench', target_budget=T_BUDGET)



    print("\nDone! Generated:")
    print("  graph1_bandwidth_latency.png")
    print("  graph2_breakdown.png")
    print("  graph3_n_latency.png")
    print("  dynamic_graph1_N_vs_Ttx.png")
    print("  dynamic_graph2_scenarios.png")
    print("  dynamic_graph3_b_heatmap.png")
    print("  dynamic_graph4_timeseries.png")