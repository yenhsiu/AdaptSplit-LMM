# LMM Split Inference 延遲測量腳本

## 系統架構概覽

```
┌─────────────────────────────────────────────────────────────────┐
│  Edge 裝置                          Cloud 端                     │
│  (RPi 4B / Jetson Nano / AGX Orin)  (GPU 伺服器)                │
│                                                                  │
│  CLIP-ViT-L                         MLP Projector               │
│  576 tokens                         + LLaVA-1.5 (Vicuna-7B)    │
│      ↓                                                           │
│  PruMerge (576 → N tokens)                                      │
│      ↓                                                           │
│  TurboQuant (量化壓縮)  ──────────────────────────────────────→ │
└─────────────────────────────────────────────────────────────────┘

T_total = T_edge + T_prumerge + T_quant + T_tx + T_cloud
```

---

## 快速開始

### 安裝相依套件

```bash
pip install torch transformers
```

> 需事先下載模型（腳本支援離線執行）：
> - Edge：`openai/clip-vit-large-patch14-336`
> - Cloud：`llava-hf/llava-1.5-7b-hf`

### 執行方式

```bash
# Jetson AGX Orin（完整測量）
python measure_latency.py --device "Jetson AGX Orin"

# Raspberry Pi 4B（記憶體較少，跳過 Cloud）
python measure_latency.py --device "Raspberry Pi 4B" --skip-cloud

# Jetson Nano（跳過 Cloud）
python measure_latency.py --device "Jetson Nano" --skip-cloud

# 僅測 Edge（跳過 T_edge ViT 載入）
python measure_latency.py --device "My Device" --skip-edge
```

### CLI 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--device` | 裝置名稱（用於輸出與 CSV 檔名） | `"Unknown Device"` |
| `--skip-edge` | 跳過 T_edge 測量（ViT 載入慢時使用） | `False` |
| `--skip-cloud` | 跳過 T_cloud 測量 | `False` |

---

## 測量項目說明

### T_edge — CLIP-ViT-L 推論時間

| 項目 | 說明 |
|------|------|
| 輸入 | `torch.randn(1, 3, 336, 336)` |
| 輸出 | 576 個 visual token（dim=1024） |
| 重複次數 | warm-up 10 次 + 計時 100 次 |
| 模型 | `openai/clip-vit-large-patch14-336` |
| Fallback | 記憶體不足時自動改用 `openai/clip-vit-base-patch16`，輸出中標註 |

### T_prumerge — Token 壓縮時間

| 項目 | 說明 |
|------|------|
| 輸入 | `(576, 1024)` 的 visual token |
| 操作 | `torch.sort()` 模擬 attention ranking，取前 N 個 |
| 輸出 | `(N, 1024)` 壓縮後 token |
| 測試 N 值 | 15, 20, 25, 30, 35, 40, 50, 60 |
| 每個 N | warm-up 10 次 + 計時 100 次 |

### T_quant — 量化時間

| 項目 | 說明 |
|------|------|
| 輸入 | `torch.randn(N, 1024).half()` (FP16) |
| 操作 | min-max 正規化後截斷至 2^B 個量化等級 |
| 測試組合 | N ∈ {15…60} × B ∈ {4, 2, 1} bit |
| 每個組合 | warm-up 10 次 + 計時 100 次 |

### T_tx — 傳輸時間（公式計算）

$$T_{tx}(ms) = \frac{N \times 1024 \times B}{S \times 10^6} \times 1000$$

| 參數 | 測試值 |
|------|--------|
| N（token 數） | 15, 20, 25, 30, 35, 40, 50, 60 |
| B（量化位元） | 1, 2, 4 bit |
| S（頻寬） | 1, 5, 20 Mbps |

> 公式計算，不進行實際網路傳輸。

### T_cloud — Cloud LLM Prefill 時間

| 項目 | 說明 |
|------|------|
| 輸入 | `torch.randn(N, 4096)`（Projector 輸出維度） |
| 模型 | `llava-hf/llava-1.5-7b-hf`（FP16） |
| Fallback | 載入失敗時改用等效 Linear layer 模擬，輸出中標註 |
| 測試 N 值 | 15, 20, 25, 30, 35, 40, 50, 60 |
| 每個 N | warm-up 10 次 + 計時 50 次 |

---

## 計時方式

```python
# 1. 排除前 10 次 warm-up
for _ in range(10):
    fn()

# 2. CUDA 同步後開始計時
torch.cuda.synchronize()          # GPU 裝置
t0 = time.perf_counter()
fn()
torch.cuda.synchronize()          # GPU 裝置
elapsed_ms = (time.perf_counter() - t0) * 1000.0

# 3. 無 GPU（Raspberry Pi）自動切換 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 輸出格式

### 終端機輸出範例

```
════════════════════════════════════════════════════════════
  裝置資訊
════════════════════════════════════════════════════════════
  裝置名稱    : Jetson AGX Orin
  GPU 型號    : NVIDIA Orin
  CUDA 版本   : 11.4
  RAM         : 32.0 GB

════════════════════════════════════════════════════════════
  T_edge  （ViT 推論）
════════════════════════════════════════════════════════════
  T_edge (clip-vit-large-patch14-336)    45.231 ±  1.203 ms

════════════════════════════════════════════════════════════
  T_prumerge  （token 壓縮）
════════════════════════════════════════════════════════════
  N=15                                    0.123 ±  0.012 ms
  N=20                                    0.145 ±  0.011 ms
  ...

════════════════════════════════════════════════════════════
  端到端延遲估算
════════════════════════════════════════════════════════════
  [最佳情況]  N=15, B=1bit, S=20Mbps
  總延遲                  XX.XXX ms
    T_edge       XX.XXX ms  ( XX.X%)
    T_prumerge   XX.XXX ms  (  X.X%)
    T_quant      XX.XXX ms  (  X.X%)
    T_tx         XX.XXX ms  (  X.X%)
    T_cloud      XX.XXX ms  ( XX.X%)
  >> 瓶頸：T_edge

  [典型情況]  N=35, B=4bit, S=5Mbps
  ...

════════════════════════════════════════════════════════════
  瓶頸分析（典型情況，N=35 B=4bit S=5Mbps）
════════════════════════════════════════════════════════════
  T_edge          XX.X%  ██████████████████████████
  T_prumerge       X.X%  ██
  T_quant          X.X%  █
  T_tx             X.X%  ████
  T_cloud         XX.X%  ████████████████
```

### CSV 輸出

檔名格式：`{device_name}_latency_results.csv`

| 欄位 | 說明 |
|------|------|
| `component` | T_edge / T_prumerge / T_quant / T_tx / T_cloud |
| `N` | token 數量 |
| `bits` | 量化位元數 |
| `bw_mbps` | 頻寬（T_tx 專用） |
| `mean_ms` | 平均延遲（ms） |
| `std_ms` | 標準差（ms） |
| `note` | 使用的模型或 fallback 標註 |

---

## 端到端估算場景

| 場景 | N | B | S (Mbps) |
|------|---|---|----------|
| 最佳情況 | 15 | 1 bit | 20 |
| 典型情況 | 35 | 4 bit | 5 |
| 最差情況 | 60 | 4 bit | 1 |

---

## Fallback 機制

```
T_edge:
  openai/clip-vit-large-patch14-336
      └─ 記憶體不足 → openai/clip-vit-base-patch16 [標註 fallback]
             └─ 仍失敗 → 回傳 FAILED，繼續執行其餘測量

T_cloud:
  llava-hf/llava-1.5-7b-hf
      └─ 載入失敗 → FakeLLM (Linear 4096→4096) [標註 fake]
```

---

## 程式碼結構

```
measure_latency.py
├── Config                  全域常數（N 值、bit 值、重複次數）
├── Helpers
│   ├── get_device()        自動偵測 CUDA / CPU
│   ├── sync()              torch.cuda.synchronize() wrapper
│   ├── timed_run()         warm-up + 計時主函式
│   └── detect_system_info() 擷取 GPU / RAM / Python 版本
├── measure_t_edge()        CLIP-ViT-L 推論測量
├── measure_t_prumerge()    Token 壓縮測量
├── simulate_quantize()     位元截斷量化模擬
├── measure_t_quant()       量化延遲測量
├── calc_t_tx()             傳輸時間公式計算
├── build_t_tx_table()      建立 N × B × S 完整組合表
├── FakeLLMPrefill          Linear layer 模擬 LLM prefill
├── measure_t_cloud()       Cloud prefill 測量
├── e2e_estimate()          端到端延遲估算 + 瓶頸判斷
├── print_results()         格式化終端機輸出
├── export_csv()            CSV 匯出
└── main()                  CLI 入口
```

---

## 目標問題對照

| 問題 | 對應測量項目 |
|------|------------|
| T_edge 在各裝置上是多少？ | `T_edge` 各裝置 CSV 比較 |
| T_cloud(N) 跟 N 是線性關係嗎？ | `T_cloud` N=15~60 曲線 |
| T_tx 跟 T_edge / T_cloud 相比大還是小？ | E2E 估算百分比欄位 |
| 瓶頸在哪裡？ | 瓶頸分析區塊（自動判斷） |
| S=5Mbps 下端到端延遲是多少？ | 典型情況（N=35, B=4bit, S=5Mbps） |

---

## 需求環境

| 項目 | 需求 |
|------|------|
| Python | 3.8+ |
| PyTorch | 任意版本（支援 CUDA 與 CPU） |
| transformers | Hugging Face transformers |
| 網路 | 離線可執行（模型需事先下載） |
| GPU | 選用（無 GPU 自動切換 CPU） |
