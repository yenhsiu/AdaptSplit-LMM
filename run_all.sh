#!/bin/bash
# run_all.sh — 全部實驗，3 GPU 平均分配（各 14 runs，序列執行）
# Usage: bash run_all.sh
# Logs: logs/gpu0.log, logs/gpu1.log, logs/gpu2.log

set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=============================="
echo "  AdaptSplit-LMM 全量實驗"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Logs → $LOG_DIR"
echo "=============================="

# ── GPU 0（14 runs）──────────────────────────────────────────
run_gpu0() {
    local LOG="$LOG_DIR/gpu0.log"
    echo "[GPU 0] 開始 $(date '+%H:%M:%S')" | tee "$LOG"

    # Exp A
    bash validate.sh --dataset pope    --budget 25600   --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope    --budget 266240  --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --budget 51200   --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --budget 589824  --cuda 0 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=1
    bash validate.sh --dataset pope --n4 0 --n2 0 --n1 25  --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope --n4 0 --n2 0 --n1 260 --cuda 0 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=2
    bash validate.sh --dataset pope --n4 0 --n2 25  --n1 0 --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope --n4 0 --n2 288 --n1 0 --cuda 0 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=4
    bash validate.sh --dataset pope --n4 28  --n2 0 --n1 0 --cuda 0 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=1
    bash validate.sh --dataset textvqa --n4 0 --n2 0 --n1 25  --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --n4 0 --n2 0 --n1 260 --cuda 0 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=2
    bash validate.sh --dataset textvqa --n4 0 --n2 25  --n1 0 --cuda 0 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --n4 0 --n2 288 --n1 0 --cuda 0 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=4
    bash validate.sh --dataset textvqa --n4 28  --n2 0 --n1 0 --cuda 0 2>&1 | tee -a "$LOG"

    echo "[GPU 0] 完成 $(date '+%H:%M:%S')" | tee -a "$LOG"
}

# ── GPU 1（14 runs）──────────────────────────────────────────
run_gpu1() {
    local LOG="$LOG_DIR/gpu1.log"
    echo "[GPU 1] 開始 $(date '+%H:%M:%S')" | tee "$LOG"

    # Exp A
    bash validate.sh --dataset pope    --budget 51200   --cuda 1 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope    --budget 589824  --cuda 1 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --budget 115500  --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — Baseline
    bash validate.sh --dataset pope    --baseline --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=1
    bash validate.sh --dataset pope --n4 0 --n2 0 --n1 50  --cuda 1 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope --n4 0 --n2 0 --n1 576 --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=2
    bash validate.sh --dataset pope --n4 0 --n2 56  --n1 0 --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=4
    bash validate.sh --dataset pope --n4 6   --n2 0 --n1 0 --cuda 1 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope --n4 65  --n2 0 --n1 0 --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=1
    bash validate.sh --dataset textvqa --n4 0 --n2 0 --n1 50  --cuda 1 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --n4 0 --n2 0 --n1 576 --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=2
    bash validate.sh --dataset textvqa --n4 0 --n2 56  --n1 0 --cuda 1 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=4
    bash validate.sh --dataset textvqa --n4 6   --n2 0 --n1 0 --cuda 1 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --n4 65  --n2 0 --n1 0 --cuda 1 2>&1 | tee -a "$LOG"

    echo "[GPU 1] 完成 $(date '+%H:%M:%S')" | tee -a "$LOG"
}

# ── GPU 2（14 runs）──────────────────────────────────────────
run_gpu2() {
    local LOG="$LOG_DIR/gpu2.log"
    echo "[GPU 2] 開始 $(date '+%H:%M:%S')" | tee "$LOG"

    # Exp A
    bash validate.sh --dataset pope    --budget 115500  --cuda 2 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --budget 25600   --cuda 2 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --budget 266240  --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — Baseline
    bash validate.sh --dataset textvqa --baseline --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=1
    bash validate.sh --dataset pope --n4 0 --n2 0 --n1 112 --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=2
    bash validate.sh --dataset pope --n4 0 --n2 12  --n1 0 --cuda 2 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope --n4 0 --n2 130 --n1 0 --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — POPE All B=4
    bash validate.sh --dataset pope --n4 12  --n2 0 --n1 0 --cuda 2 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset pope --n4 144 --n2 0 --n1 0 --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=1
    bash validate.sh --dataset textvqa --n4 0 --n2 0 --n1 112 --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=2
    bash validate.sh --dataset textvqa --n4 0 --n2 12  --n1 0 --cuda 2 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --n4 0 --n2 130 --n1 0 --cuda 2 2>&1 | tee -a "$LOG"

    # Exp B — TextVQA All B=4
    bash validate.sh --dataset textvqa --n4 12  --n2 0 --n1 0 --cuda 2 2>&1 | tee -a "$LOG"
    bash validate.sh --dataset textvqa --n4 144 --n2 0 --n1 0 --cuda 2 2>&1 | tee -a "$LOG"

    echo "[GPU 2] 完成 $(date '+%H:%M:%S')" | tee -a "$LOG"
}

# ── 三張 GPU 同時啟動 ──────────────────────────────────────
cd "$SCRIPT_DIR"

run_gpu0 &
PID0=$!

run_gpu1 &
PID1=$!

run_gpu2 &
PID2=$!

echo "GPU 0 PID: $PID0"
echo "GPU 1 PID: $PID1"
echo "GPU 2 PID: $PID2"
echo "所有 job 已在背景執行，可以去睡覺了"
echo "查看進度："
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu1.log"
echo "  tail -f $LOG_DIR/gpu2.log"

wait $PID0 $PID1 $PID2

echo ""
echo "=============================="
echo "  全部完成！$(date '+%Y-%m-%d %H:%M:%S')"
echo "  結果請看各 log 檔"
echo "=============================="
