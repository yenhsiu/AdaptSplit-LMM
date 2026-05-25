"""
動的 B 制御の有効性シミュレーション
固定 B=4 vs 固定 B=1 vs 動的 b* の比較

現在のロジック：
    b* = max B ∈ {4, 2, 1}
    条件：T_tx(N, B, S) ≤ T_budget
    → T_budget 内で最も高い B を選ぶ
    → これは自動的に acc も最大化している
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# 設定
# ============================================================

TOKEN_DIM = 1024
T_EDGE    = 231.5
T_BUDGET  = T_EDGE

# POPE の acc データ（PruMerge plus）
ACC_POPE = {4: 84.5, 2: 84.1, 1: 83.7}
ACC_MME  = {4: 1426.15, 2: 1419.98, 1: 1371.43}

# ============================================================
# T_tx 計算・動的 b* 選択
# ============================================================

def T_tx(N, B, S):
    return N * TOKEN_DIM * B / (S * 1e6) * 1000

def dynamic_b(N, S, budget=T_BUDGET):
    """
    T_budget 内で最も高い B を選ぶ
    → 自動的に acc も最大化される
    """
    for B in [4, 2, 1]:
        if T_tx(N, B, S) <= budget:
            return B
    return 1

# ============================================================
# Graph 1：N vs T_tx（上：遅延，下：b* と acc）
# ============================================================

def plot_n_vs_ttx(S_target=5, save_path="dynamic_graph1_N_vs_Ttx.png"):
    N_vals = np.arange(10, 351, 5)

    ttx_b4  = [T_tx(N, 4, S_target) for N in N_vals]
    ttx_b2  = [T_tx(N, 2, S_target) for N in N_vals]
    ttx_b1  = [T_tx(N, 1, S_target) for N in N_vals]
    ttx_dyn = [T_tx(N, dynamic_b(N, S_target), S_target) for N in N_vals]
    b_chosen = [dynamic_b(N, S_target) for N in N_vals]
    acc_dyn  = [ACC_POPE[b] for b in b_chosen]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8),
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    # --- T_tx 比較 ---
    ax = axes[0]
    ax.plot(N_vals, ttx_b4,  color='#A32D2D', linewidth=2,
            label="Fixed B=4", linestyle='-')
    ax.plot(N_vals, ttx_b2,  color='#BA7517', linewidth=2,
            label="Fixed B=2", linestyle='-.')
    ax.plot(N_vals, ttx_b1,  color='#888780', linewidth=2,
            label="Fixed B=1", linestyle='--')
    ax.plot(N_vals, ttx_dyn, color='#1D9E75', linewidth=2.5,
            label="Dynamic b* (proposed)", linestyle='-')
    ax.axhline(y=T_BUDGET, color='black', linewidth=1.5,
               linestyle=':', label=f"T_budget = {T_BUDGET:.0f}ms")
    ax.axvspan(10,  100, alpha=0.05, color='#378ADD')
    ax.axvspan(30,  350, alpha=0.05, color='#BA7517')
    ax.axvline(x=32,  color='#378ADD', linewidth=1, linestyle=':',
               alpha=0.7, label="base avg N=32")
    ax.axvline(x=145, color='#BA7517', linewidth=1, linestyle=':',
               alpha=0.7, label="plus avg N=145")
    ax.set_ylabel("T_tx (ms)", fontsize=12)
    ax.set_title(f"Token Count N vs T_tx  (S={S_target} Mbps)\n"
                 f"Dynamic b* stays within T_budget while maximizing acc",
                 fontsize=11)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 350)

    # --- 選ばれた b* ---
    ax2 = axes[1]
    ax2.step(N_vals, b_chosen, color='#1D9E75', linewidth=2, where='post')
    ax2.fill_between(N_vals, b_chosen, step='post',
                     alpha=0.2, color='#1D9E75')
    ax2.set_yticks([1, 2, 4])
    ax2.set_yticklabels(['B=1', 'B=2', 'B=4'])
    ax2.set_ylabel("b*", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 350)

    # --- 対応する POPE acc ---
    ax3 = axes[2]
    ax3.step(N_vals, acc_dyn, color='#378ADD', linewidth=2, where='post')
    ax3.fill_between(N_vals, acc_dyn, step='post',
                     alpha=0.2, color='#378ADD')
    ax3.set_ylim(83, 85.5)
    ax3.set_ylabel("POPE acc (%)", fontsize=10)
    ax3.set_xlabel("Number of Visual Tokens N", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(10, 350)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 2：シナリオ比較（T_tx + acc の 2 軸）
# ============================================================

def plot_scenarios(save_path="dynamic_graph2_scenarios.png"):
    scenarios = [
        {"label": "S1\nN=30\nS=5Mbps",   "N": 30,  "S": 5.0},
        {"label": "S2\nN=145\nS=5Mbps",  "N": 145, "S": 5.0},
        {"label": "S3\nN=350\nS=5Mbps",  "N": 350, "S": 5.0},
        {"label": "S4\nN=145\nS=1Mbps",  "N": 145, "S": 1.0},
        {"label": "S5\nN=350\nS=1Mbps",  "N": 350, "S": 1.0},
        {"label": "S6\nN=100\nS=0.5Mbps","N": 100, "S": 0.5},
    ]

    x = np.arange(len(scenarios))
    width = 0.18  # 4本並べるので少し細くする

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={'height_ratios': [2, 1]})

    # --- T_tx 比較（4本並べる）---
    # 位置：-1.5w, -0.5w, +0.5w, +1.5w
    offsets = [-1.5, -0.5, 0.5, 1.5]
    configs = [
        (4, '#A32D2D', 'Fixed B=4'),
        (2, '#BA7517', 'Fixed B=2'),
        (1, '#888780', 'Fixed B=1'),
    ]

    for (B, color, label), offset in zip(configs, offsets[:3]):
        vals = [T_tx(s["N"], B, s["S"]) for s in scenarios]
        ax1.bar(x + offset * width, vals, width,
                label=label, color=color, alpha=0.85)

    # 動的 b*（4本目）
    dyn_vals = [T_tx(s["N"], dynamic_b(s["N"], s["S"]), s["S"])
                for s in scenarios]
    b_chosen = [dynamic_b(s["N"], s["S"]) for s in scenarios]
    ax1.bar(x + offsets[3] * width, dyn_vals, width,
            label="Dynamic b*", color='#1D9E75', alpha=0.9)

    ax1.axhline(y=T_BUDGET, color='black', linewidth=1.5,
                linestyle=':', label=f"T_budget={T_BUDGET:.0f}ms")

    # アノテーション（B=4 と Dynamic b* のみ）
    b4_vals = [T_tx(s["N"], 4, s["S"]) for s in scenarios]
    for i, (b4, dyn, b) in enumerate(zip(b4_vals, dyn_vals, b_chosen)):
        ax1.text(i + offsets[0] * width, b4 + 15, f"{b4:.0f}",
                 ha='center', va='bottom', fontsize=7.5, color='#A32D2D')
        ax1.text(i + offsets[3] * width, dyn + 15,
                 f"{dyn:.0f}\nB={b}",
                 ha='center', va='bottom', fontsize=7.5, color='#1D9E75')

    ax1.set_xticks(x)
    ax1.set_xticklabels([s["label"] for s in scenarios], fontsize=9)
    ax1.set_ylabel("T_tx (ms)", fontsize=11)
    ax1.set_title("T_tx Comparison: Fixed B=4 / B=2 / B=1 vs Dynamic b*",
                  fontsize=11)
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, axis='y', alpha=0.3)

    # --- acc 比較 ---
    acc_dyn = [ACC_POPE[b] for b in b_chosen]

    ax2.plot(x, [ACC_POPE[4]] * len(scenarios), 'o--',
             color='#A32D2D', linewidth=1.5,
             label=f"Fixed B=4 ({ACC_POPE[4]}%)")
    ax2.plot(x, [ACC_POPE[2]] * len(scenarios), 's--',
             color='#BA7517', linewidth=1.5,
             label=f"Fixed B=2 ({ACC_POPE[2]}%)")
    ax2.plot(x, [ACC_POPE[1]] * len(scenarios), '^--',
             color='#888780', linewidth=1.5,
             label=f"Fixed B=1 ({ACC_POPE[1]}%)")
    ax2.plot(x, acc_dyn, 'D-',
             color='#1D9E75', linewidth=2.5, markersize=8,
             label="Dynamic b* (max acc within budget)")

    # b* の値をアノテーション
    for i, (b, acc) in enumerate(zip(b_chosen, acc_dyn)):
        ax2.annotate(f"B={b}", (i, acc),
                     textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=8, color='#1D9E75')

    ax2.set_xticks(x)
    ax2.set_xticklabels([s["label"] for s in scenarios], fontsize=9)
    ax2.set_ylabel("POPE Acc (%)", fontsize=11)
    ax2.set_ylim(83, 85.5)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 3：b* ヒートマップ
# ============================================================

def plot_b_heatmap(save_path="dynamic_graph3_b_heatmap.png"):
    N_vals = np.arange(10, 351, 10)
    S_vals = [0.5, 1, 2, 5, 10, 20]

    b_matrix = np.zeros((len(S_vals), len(N_vals)))
    for i, S in enumerate(S_vals):
        for j, N in enumerate(N_vals):
            b_matrix[i, j] = dynamic_b(N, S)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # b* ヒートマップ
    cmap = plt.cm.get_cmap('RdYlGn', 3)
    im = ax1.imshow(b_matrix, aspect='auto', cmap=cmap,
                    vmin=0.5, vmax=4.5,
                    extent=[N_vals[0], N_vals[-1], len(S_vals)-0.5, -0.5])
    ax1.set_yticks(range(len(S_vals)))
    ax1.set_yticklabels([f"{s} Mbps" for s in S_vals])
    ax1.set_xlabel("N (tokens)", fontsize=11)
    ax1.set_ylabel("S (Bandwidth)", fontsize=11)
    ax1.set_title("Dynamic b* Selection Map\n"
                  "Green=B=4 (high acc), Yellow=B=2, Red=B=1", fontsize=10)
    ax1.axvline(x=32,  color='white', linewidth=2, linestyle='--')
    ax1.axvline(x=145, color='white', linewidth=2, linestyle='--')
    cbar = plt.colorbar(im, ax=ax1, ticks=[1, 2, 4])
    cbar.ax.set_yticklabels(['B=1', 'B=2', 'B=4'])

    # acc ヒートマップ
    acc_matrix = np.vectorize(lambda b: ACC_POPE[b])(b_matrix.astype(int))
    im2 = ax2.imshow(acc_matrix, aspect='auto',
                     cmap='RdYlGn',
                     vmin=83.5, vmax=84.7,
                     extent=[N_vals[0], N_vals[-1], len(S_vals)-0.5, -0.5])
    ax2.set_yticks(range(len(S_vals)))
    ax2.set_yticklabels([f"{s} Mbps" for s in S_vals])
    ax2.set_xlabel("N (tokens)", fontsize=11)
    ax2.set_title("POPE Accuracy with Dynamic b*\n"
                  "Darker = higher acc", fontsize=10)
    ax2.axvline(x=32,  color='white', linewidth=2, linestyle='--')
    ax2.axvline(x=145, color='white', linewidth=2, linestyle='--')
    plt.colorbar(im2, ax=ax2, label="POPE Acc (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 4：時系列シミュレーション
# ============================================================

def plot_timeseries(save_path="dynamic_graph4_timeseries.png"):
    np.random.seed(42)
    n_steps = 30

    N_series = np.clip(
        np.random.normal(145, 60, n_steps).astype(int), 30, 350)
    S_series = np.clip(
        np.random.normal(3, 2, n_steps), 0.3, 15)

    ttx_b4  = [T_tx(N, 4, S) for N, S in zip(N_series, S_series)]
    ttx_b1  = [T_tx(N, 1, S) for N, S in zip(N_series, S_series)]
    ttx_dyn = [T_tx(N, dynamic_b(N, S), S) for N, S in zip(N_series, S_series)]
    b_chosen = [dynamic_b(N, S) for N, S in zip(N_series, S_series)]
    acc_dyn  = [ACC_POPE[b] for b in b_chosen]

    steps = np.arange(n_steps)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9),
                              gridspec_kw={'height_ratios': [1, 1, 2, 1]})

    # N の変動
    axes[0].plot(steps, N_series, color='#BA7517', linewidth=1.5)
    axes[0].axhline(y=32,  color='#378ADD', linewidth=1,
                    linestyle='--', alpha=0.7, label='base avg N=32')
    axes[0].axhline(y=145, color='#BA7517', linewidth=1,
                    linestyle='--', alpha=0.7, label='plus avg N=145')
    axes[0].set_ylabel("N (tokens)", fontsize=10)
    axes[0].set_title("Simulated N and S variation with Dynamic b* control",
                      fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, n_steps-1)

    # S の変動
    axes[1].plot(steps, S_series, color='#888780', linewidth=1.5)
    axes[1].set_ylabel("S (Mbps)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, n_steps-1)

    # T_tx 比較
    axes[2].plot(steps, ttx_b4,  color='#A32D2D', linewidth=2,
                 label="Fixed B=4", alpha=0.9)
    axes[2].plot(steps, ttx_b1,  color='#888780', linewidth=1.5,
                 label="Fixed B=1", alpha=0.7, linestyle='--')
    axes[2].plot(steps, ttx_dyn, color='#1D9E75', linewidth=2.5,
                 label="Dynamic b*", alpha=0.9)
    axes[2].axhline(y=T_BUDGET, color='black', linewidth=1.5,
                    linestyle=':', label=f"T_budget={T_BUDGET:.0f}ms")

    for i, t in enumerate(ttx_b4):
        if t > T_BUDGET:
            axes[2].axvspan(i-0.4, i+0.4, alpha=0.15, color='red')

    over_b4  = sum(1 for t in ttx_b4 if t > T_BUDGET)
    over_dyn = sum(1 for t in ttx_dyn if t > T_BUDGET)
    axes[2].text(0.98, 0.95,
                 f"Exceeded budget:\nFixed B=4: {over_b4}/{n_steps}\n"
                 f"Dynamic b*: {over_dyn}/{n_steps}",
                 transform=axes[2].transAxes, ha='right', va='top',
                 fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_ylabel("T_tx (ms)", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, n_steps-1)

    # acc の変動（動的 b* で選ばれた B の acc）
    axes[3].step(steps, acc_dyn, color='#378ADD', linewidth=2, where='post')
    axes[3].fill_between(steps, acc_dyn, step='post',
                         alpha=0.2, color='#378ADD')
    axes[3].axhline(y=ACC_POPE[4], color='#A32D2D', linewidth=1,
                    linestyle='--', label=f"B=4 acc ({ACC_POPE[4]}%)")
    axes[3].axhline(y=ACC_POPE[1], color='#888780', linewidth=1,
                    linestyle='--', label=f"B=1 acc ({ACC_POPE[1]}%)")
    axes[3].set_ylim(83, 85.5)
    axes[3].set_ylabel("POPE Acc (%)", fontsize=10)
    axes[3].set_xlabel("Time Step", fontsize=10)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, n_steps-1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating dynamic B control simulation graphs...")

    print("\n[1/4] N vs T_tx")
    plot_n_vs_ttx(S_target=5)

    print("[2/4] Scenario comparison (T_tx + acc)")
    plot_scenarios()

    print("[3/4] b* heatmap + acc heatmap")
    plot_b_heatmap()

    print("[4/4] Time series simulation")
    plot_timeseries()

    print("\nDone!")

    # サマリー
    print(f"\n{'N':>5} {'S':>6} {'b*':>4} "
          f"{'T_tx_B4':>10} {'T_tx_B2':>10} "
          f"{'T_tx_dyn':>10} {'POPE':>8}")
    print("-" * 60)
    for N, S in [(32,5),(145,5),(350,5),(145,1),(350,1),(100,0.5)]:
        b = dynamic_b(N, S)
        print(f"{N:>5} {S:>5.1f}M {b:>4} "
              f"{T_tx(N,4,S):>9.1f}ms "
              f"{T_tx(N,2,S):>9.1f}ms "
              f"{T_tx(N,b,S):>9.1f}ms "
              f"{ACC_POPE[b]:>7.1f}%")

# ============================================================
# Graph 1：N が変動する場合の T_tx 比較
# ============================================================

def plot_n_vs_ttx(S_target=5, save_path="dynamic_graph1_N_vs_Ttx.png"):
    """
    X軸：token 数 N（10〜350）
    Y軸：T_tx（ms）
    固定 B=4 / 固定 B=1 / 動的 b* の比較
    S を固定してNの変動の影響を示す
    """
    N_vals = np.arange(10, 351, 5)

    ttx_b4   = [T_tx(N, 4, S_target) for N in N_vals]
    ttx_b1   = [T_tx(N, 1, S_target) for N in N_vals]
    ttx_dyn  = [T_tx(N, dynamic_b(N, S_target), S_target) for N in N_vals]
    b_chosen = [dynamic_b(N, S_target) for N in N_vals]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # T_tx の比較
    ax1.plot(N_vals, ttx_b4,  color='#A32D2D', linewidth=2,
             label="Fixed B=4", linestyle='-')
    ax1.plot(N_vals, ttx_b1,  color='#888780', linewidth=2,
             label="Fixed B=1", linestyle='--')
    ax1.plot(N_vals, ttx_dyn, color='#1D9E75', linewidth=2.5,
             label="Dynamic b*", linestyle='-')

    # T_budget ライン
    ax1.axhline(y=T_BUDGET, color='black', linewidth=1.5,
                linestyle=':', label=f"T_budget = {T_BUDGET:.0f}ms (T_edge)")

    # PruMerge base / plus の平均 N
    ax1.axvline(x=32,  color='#378ADD', linewidth=1, linestyle=':',
                alpha=0.7, label="PruMerge base avg (N=32)")
    ax1.axvline(x=145, color='#BA7517', linewidth=1, linestyle=':',
                alpha=0.7, label="PruMerge plus avg (N=145)")

    # 変動範囲のシェーディング
    ax1.axvspan(10, 100, alpha=0.05, color='#378ADD', label="base range (10–100)")
    ax1.axvspan(30, 350, alpha=0.05, color='#BA7517', label="plus range (30–350)")

    ax1.set_ylabel("T_tx (ms)", fontsize=12)
    ax1.set_title(f"Token Count N vs T_tx  (S={S_target} Mbps)\n"
                  f"Fixed B=4 explodes at large N; Dynamic b* stays within budget",
                  fontsize=11)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10, 350)

    # 選ばれた b* を下段に表示
    ax2.step(N_vals, b_chosen, color='#1D9E75', linewidth=2, where='post')
    ax2.set_yticks([1, 2, 4])
    ax2.set_yticklabels(['B=1', 'B=2', 'B=4'])
    ax2.set_xlabel("Number of Visual Tokens N", fontsize=12)
    ax2.set_ylabel("Selected b*", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 350)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 2：シナリオ比較（N と S が同時に変動）
# ============================================================

def plot_scenarios(save_path="dynamic_graph2_scenarios.png"):
    """
    代表的なシナリオで固定 B vs 動的 b* の T_tx を比較
    """
    scenarios = [
        {"label": "Scenario 1\nN=30\nS=5Mbps",  "N": 30,  "S": 5.0},
        {"label": "Scenario 2\nN=145\nS=5Mbps", "N": 145, "S": 5.0},
        {"label": "Scenario 3\nN=350\nS=5Mbps", "N": 350, "S": 5.0},
        {"label": "Scenario 4\nN=145\nS=1Mbps", "N": 145, "S": 1.0},
        {"label": "Scenario 5\nN=350\nS=1Mbps", "N": 350, "S": 1.0},
        {"label": "Scenario 6\nN=100\nS=0.5Mbps","N": 100, "S": 0.5},
    ]

    labels   = [s["label"] for s in scenarios]
    ttx_b4   = [T_tx(s["N"], 4, s["S"]) for s in scenarios]
    ttx_b1   = [T_tx(s["N"], 1, s["S"]) for s in scenarios]
    ttx_dyn  = [T_tx(s["N"], dynamic_b(s["N"], s["S"]), s["S"]) for s in scenarios]
    b_chosen = [dynamic_b(s["N"], s["S"]) for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_b4  = ax.bar(x - width, ttx_b4,  width, label="Fixed B=4",
                      color='#A32D2D', alpha=0.85)
    bars_b1  = ax.bar(x,          ttx_b1,  width, label="Fixed B=1",
                      color='#888780', alpha=0.85)
    bars_dyn = ax.bar(x + width,  ttx_dyn, width, label="Dynamic b*",
                      color='#1D9E75', alpha=0.85)

    # T_budget ライン
    ax.axhline(y=T_BUDGET, color='black', linewidth=1.5,
               linestyle=':', label=f"T_budget = {T_BUDGET:.0f}ms")

    # 値のアノテーション（固定 B=4 と動的 b* のみ）
    for i, (b4, dyn, b) in enumerate(zip(ttx_b4, ttx_dyn, b_chosen)):
        # 固定 B=4
        ax.text(i - width, b4 + 10, f"{b4:.0f}ms",
                ha='center', va='bottom', fontsize=7.5, color='#A32D2D')
        # 動的 b*
        ax.text(i + width, dyn + 10, f"{dyn:.0f}ms\n(B={b})",
                ha='center', va='bottom', fontsize=7.5, color='#1D9E75')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("T_tx (ms)", fontsize=12)
    ax.set_title("T_tx Comparison: Fixed B=4 vs Fixed B=1 vs Dynamic b*\n"
                 "Dynamic b* keeps T_tx within budget across all scenarios",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 3：N × S の全組み合わせでの b* ヒートマップ
# ============================================================

def plot_b_heatmap(save_path="dynamic_graph3_b_heatmap.png"):
    """
    X軸：N（10〜350）
    Y軸：S（0.5〜20 Mbps）
    色：動的 b* の値（1, 2, 4）
    """
    N_vals = np.arange(10, 351, 10)
    S_vals = [0.5, 1, 2, 5, 10, 20]

    b_matrix = np.zeros((len(S_vals), len(N_vals)))
    for i, S in enumerate(S_vals):
        for j, N in enumerate(N_vals):
            b_matrix[i, j] = dynamic_b(N, S)

    fig, ax = plt.subplots(figsize=(10, 4))

    cmap = plt.cm.get_cmap('RdYlGn', 3)
    im = ax.imshow(b_matrix, aspect='auto', cmap=cmap,
                   vmin=0.5, vmax=4.5,
                   extent=[N_vals[0], N_vals[-1],
                           len(S_vals)-0.5, -0.5])

    ax.set_yticks(range(len(S_vals)))
    ax.set_yticklabels([f"{s} Mbps" for s in S_vals], fontsize=10)
    ax.set_xlabel("Number of Visual Tokens N", fontsize=12)
    ax.set_ylabel("Bandwidth S", fontsize=12)
    ax.set_title("Dynamic b* Selection Map\n"
                 "Green=B=4 (high quality), Yellow=B=2, Red=B=1 (speed priority)",
                 fontsize=11)

    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 4])
    cbar.set_label("Selected b*", fontsize=10)
    cbar.ax.set_yticklabels(['B=1', 'B=2', 'B=4'])

    # PruMerge base / plus の平均 N
    ax.axvline(x=32,  color='#378ADD', linewidth=2, linestyle='--',
               label='base avg N=32')
    ax.axvline(x=145, color='#BA7517', linewidth=2, linestyle='--',
               label='plus avg N=145')
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Graph 4：時系列シミュレーション（N と S が変動する場合）
# ============================================================

def plot_timeseries(save_path="dynamic_graph4_timeseries.png"):
    """
    時間軸で N と S が変動する場合の T_total シミュレーション
    固定 B=4 vs 動的 b* の比較
    """
    # シミュレーションシナリオ（時系列）
    np.random.seed(42)
    n_steps = 30

    # N の変動（PruMerge plus の現実的な範囲）
    N_series = np.clip(
        np.random.normal(145, 60, n_steps).astype(int),
        30, 350
    )

    # S の変動（モバイルネットワーク）
    S_series = np.clip(
        np.random.normal(3, 2, n_steps),
        0.3, 15
    )

    # T_tx の計算
    ttx_b4  = [T_tx(N, 4, S) for N, S in zip(N_series, S_series)]
    ttx_b1  = [T_tx(N, 1, S) for N, S in zip(N_series, S_series)]
    ttx_dyn = [T_tx(N, dynamic_b(N, S), S) for N, S in zip(N_series, S_series)]

    steps = np.arange(n_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8),
                              gridspec_kw={'height_ratios': [1, 1, 2]})

    # N の変動
    axes[0].plot(steps, N_series, color='#BA7517', linewidth=1.5)
    axes[0].axhline(y=32,  color='#378ADD', linewidth=1, linestyle='--',
                    alpha=0.7, label='base avg')
    axes[0].axhline(y=145, color='#BA7517', linewidth=1, linestyle='--',
                    alpha=0.7, label='plus avg')
    axes[0].set_ylabel("N (tokens)", fontsize=10)
    axes[0].set_title("Simulated N and S variation over time", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, n_steps-1)

    # S の変動
    axes[1].plot(steps, S_series, color='#888780', linewidth=1.5)
    axes[1].set_ylabel("S (Mbps)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, n_steps-1)

    # T_tx の比較
    axes[2].plot(steps, ttx_b4,  color='#A32D2D', linewidth=2,
                 label="Fixed B=4", alpha=0.9)
    axes[2].plot(steps, ttx_b1,  color='#888780', linewidth=1.5,
                 label="Fixed B=1", alpha=0.7, linestyle='--')
    axes[2].plot(steps, ttx_dyn, color='#1D9E75', linewidth=2.5,
                 label="Dynamic b*", alpha=0.9)
    axes[2].axhline(y=T_BUDGET, color='black', linewidth=1.5,
                    linestyle=':', label=f"T_budget={T_BUDGET:.0f}ms")

    # T_budget を超えた部分を強調
    for i, t in enumerate(ttx_b4):
        if t > T_BUDGET:
            axes[2].axvspan(i-0.4, i+0.4, alpha=0.15, color='red')

    axes[2].set_ylabel("T_tx (ms)", fontsize=10)
    axes[2].set_xlabel("Time Step", fontsize=10)
    axes[2].set_title("Fixed B=4 exceeds budget (red shading) when N×S is large;\n"
                       "Dynamic b* stays within budget", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, n_steps-1)

    # 統計を表示
    over_budget_b4  = sum(1 for t in ttx_b4 if t > T_BUDGET)
    over_budget_dyn = sum(1 for t in ttx_dyn if t > T_BUDGET)
    axes[2].text(0.98, 0.95,
                 f"Exceeded budget:\n"
                 f"Fixed B=4: {over_budget_b4}/{n_steps} steps\n"
                 f"Dynamic b*: {over_budget_dyn}/{n_steps} steps",
                 transform=axes[2].transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating dynamic B control simulation graphs...")

    print("\n[1/4] N vs T_tx (S=5Mbps fixed)")
    plot_n_vs_ttx(S_target=5)

    print("[2/4] Scenario comparison")
    plot_scenarios()

    print("[3/4] b* heatmap (N × S)")
    plot_b_heatmap()

    print("[4/4] Time series simulation")
    plot_timeseries()

    print("\nDone! Generated:")
    print("  dynamic_graph1_N_vs_Ttx.png")
    print("  dynamic_graph2_scenarios.png")
    print("  dynamic_graph3_b_heatmap.png")
    print("  dynamic_graph4_timeseries.png")

    # サマリー
    print("\n=== Dynamic b* Summary ===")
    test_cases = [
        (32,  5.0), (145, 5.0), (350, 5.0),
        (32,  1.0), (145, 1.0), (350, 1.0),
        (100, 0.5),
    ]
    print(f"{'N':>6} {'S':>8} {'b*':>5} {'T_tx_b4':>12} {'T_tx_dyn':>12}")
    print("-" * 50)
    for N, S in test_cases:
        b = dynamic_b(N, S)
        print(f"{N:>6} {S:>7.1f}M {b:>5} "
              f"{T_tx(N, 4, S):>11.1f}ms "
              f"{T_tx(N, b, S):>11.1f}ms")