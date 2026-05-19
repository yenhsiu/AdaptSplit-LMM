# LMM Split Inference 延遲測量腳本設計

## 比較手法一覧

| 手法 | 説明 | Edge 処理 | 送信データ | Cloud 処理 |
|------|------|-----------|-----------|-----------|
| ① Edge Inference（original） | 全部 Edge で実行 | LLaVA-1.5-7b（ViT + Projector + LLM） | なし | なし |
| ② Edge Inference + PruMerge | 全部 Edge + PruMerge | ViT + PruMerge + Projector + LLM | なし | なし |
| ③ Cloud Inference（original） | 生画像を送信 | なし | 生画像（2.71Mb） | LLaVA-1.5-7b（ViT + Projector + LLM） |
| ④ Cloud Inference + PruMerge | 生画像を送信 | なし | 生画像（2.71Mb） | ViT + PruMerge + Projector + LLM |
| ⑤ Split + 量化のみ（提案） | PruMerge なし，量化して送信 | ViT + TurboQuant | 量化 tokens（N=576） | Projector + LLM |
| ⑥ Split + PruMerge base + TurboQuant（提案） | N≈36 に削減して量化 | ViT + PruMerge(base) + TurboQuant | 量化 tokens（N≈36） | Projector + LLM |
| ⑦ Split + PruMerge plus + TurboQuant（提案） | N≈145 に削減して量化 | ViT + PruMerge(plus) + TurboQuant | 量化 tokens（N≈145） | Projector + LLM |

---

## システム構成

```
┌─────────────────────────────────────────────────────────────────┐
│  Edge 裝置（Jetson AGX Orin）        Cloud 端（GPU サーバー）   │
│                                                                  │
│  CLIP-ViT-L（576 tokens）  ──→  MLP Projector                  │
│      ↓                           + LLaVA-1.5（Vicuna-7B）      │
│  PruMerge（576 → N tokens）                                     │
│      ↓                                                           │
│  TurboQuant MSECompressor  ────────────────────────────────────→│
└─────────────────────────────────────────────────────────────────┘

T_total = T_edge + T_prumerge + T_quant + T_tx + T_cloud
```

---


## ファイル構成
```
measure_edge.py              → results_edge.csv       （⑤⑥⑦ の Edge 側コンポーネント実測）
measure_cloud.py             → results_cloud.csv      （⑤⑥⑦ の Cloud 側コンポーネント実測）
measure_latency_ori.py       → results_latency_ori.csv（①③ の整個 model を同一 device で実測）(デバイス名を指定)
measure_latency_prumerge.py  → results_latency_pm.csv （②④ の整個 model を同一 device で実測）(デバイス名を指定)
```
 
③④ の T_total（送信込み）と ⑤⑥⑦ の T_total は result_analysis で計算する。
 
---

## measure_edge.py
 
**目的：** ⑤⑥⑦ の Edge 側コンポーネントの処理時間を実測する。
 
**測定項目と用途：**
 
| 項目 | 用途 | 回数 |
|------|------|------|
| T_edge | ViT のみ（手法⑤⑥⑦ 用） | warm-up 10 + 計測 100 |
| T_prumerge_base | PruMerge base のみ（手法⑥ 用） | warm-up 10 + 計測 100 |
| T_prumerge_plus | PruMerge plus のみ（手法⑦ 用） | warm-up 10 + 計測 100 |
| T_quant(N, B) | TurboQuant MSECompressor（手法⑤⑥⑦ 用） | warm-up 10 + 計測 100 |
 
**T_quant の測定条件：**
- N = 36, 145, 576
- B = 1, 2, 4
- 合計 9 組み合わせ
**実装の注意：**
- T_prumerge は実際の PruMerge コードを使用（torch.sort() 模擬は不可）
- T_quant は実際の MSECompressor を使用（min-max 模擬は不可）
  ```python
  from turboquant.compressors_v3 import MSECompressor
  compressor = MSECompressor(head_dim=1024, bits=B, seed=42, device="cuda")
  ```
- 全測定で `torch.cuda.synchronize()` を使用
**出力フォーマット（results_edge.csv）：**
 
```
component, N, B, mean_ms, std_ms
T_edge, -, -, ?, ?
T_prumerge_base, -, -, ?, ?
T_prumerge_plus, -, -, ?, ?
T_quant, 36, 4, ?, ?
T_quant, 36, 2, ?, ?
T_quant, 36, 1, ?, ?
T_quant, 145, 4, ?, ?
T_quant, 145, 2, ?, ?
T_quant, 145, 1, ?, ?
T_quant, 576, 4, ?, ?
T_quant, 576, 2, ?, ?
T_quant, 576, 1, ?, ?
```
 
---
 
## measure_cloud.py
 
**目的：** ⑤⑥⑦ の Cloud 側（Projector + LLM）の処理時間を実測する。
 
**測定項目と用途：**
 
| 項目 | 用途 | 回数 |
|------|------|------|
| T_cloud(N) | Projector + LLM のみ（手法⑤⑥⑦ 用） | warm-up 10 + 計測 50 |
 
**T_cloud の測定条件：**
- N = 36, 145, 576
- 入力：`torch.randn(N, 1024).half()`
- text tokens（約 40 個）を含めた状態で LLM に入力
**出力フォーマット（results_cloud.csv）：**
 
```
component, N, mean_ms, std_ms
T_cloud, 36, ?, ?
T_cloud, 145, ?, ?
T_cloud, 576, ?, ?
```
 
---
 
## measure_latency_ori.py
 
**目的：** ①③ の整個 model を同一 device で実測する。
 
**測定項目と用途：**
 
| 項目 | 用途 | 回数 |
|------|------|------|
| T_latency_edge_ori | ViT + Projector + LLM の合計（手法① 用） | warm-up 10 + 計測 20 |
| T_latency_cloud_ori | ViT + Projector + LLM の合計（手法③ 用，同 device で実測） | warm-up 10 + 計測 20 |
 
**注意：**
- ①③ は同じ model を同じ device で実行するため T_latency_edge_ori = T_latency_cloud_ori
- ③ の T_total（送信込み）は result_analysis で T_tx_image を加算して計算する
**出力フォーマット（results_latency_ori.csv）：**
 
```
component, mean_ms, std_ms
T_latency_ori, ?, ?
```
 
---
 
## measure_latency_prumerge.py
 
**目的：** ②④ の整個 model（PruMerge あり）を同一 device で実測する。
 
**測定項目と用途：**
 
| 項目 | 用途 | 回数 |
|------|------|------|
| T_latency_prumerge_base | ViT + PruMerge(base) + Projector + LLM（手法②④ 用） | warm-up 10 + 計測 20 |
| T_latency_prumerge_plus | ViT + PruMerge(plus) + Projector + LLM（手法②④ 用） | warm-up 10 + 計測 20 |
 
**注意：**
- ②④ は同じ model（PruMerge あり）を同じ device で実行
- ④ の T_total（送信込み）は result_analysis で T_tx_image を加算して計算する
**出力フォーマット（results_latency_pm.csv）：**
 
```
component, mean_ms, std_ms
T_latency_prumerge_base, ?, ?
T_latency_prumerge_plus, ?, ?
```
 
---
 
## 計時方式（全ファイル共通）
 
```python
# GPU warm-up
for _ in range(10):
    fn()
 
# 計測
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_TRIALS):
    fn()
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - t0) / N_TRIALS * 1000.0
```
 
**実装の注意：**
- T_prumerge は実際の PruMerge コードを使用（torch.sort() 模擬は不可）
- T_quant は実際の MSECompressor を使用（min-max 模擬は不可）
  ```python
  from turboquant.compressors_v3 import MSECompressor
  compressor = MSECompressor(head_dim=1024, bits=B, seed=42, device="cuda")
  ```
- 全測定で `torch.cuda.synchronize()` を使用