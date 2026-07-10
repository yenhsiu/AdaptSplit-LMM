#!/bin/bash
# run_pipeline.sh — Search then validate
#
# Usage:
#   bash run_pipeline.sh [options]
#
# Options:
#   --budgets       Budget list (default: 115500 266240 589824)
#   --n-trials      Optuna trials per budget (default: 50)
#   --datasets      Validation datasets (default: pope textvqa mmbench)
#   --cuda          GPU id (default: 0)
#   --merge         Use prune+merge token selection
#   --mme-opt-only  Validate only MME-opt configs (skip All B=1/2/4)
#   --output        Path to save search result JSON
#   --seed          Random seed for Optuna TPE sampler (default: 42)

set -e

PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# ── Defaults ──
BUDGETS="115500 266240 589824"
N_TRIALS=50
DATASETS="pope textvqa mmbench"
CUDA=0
MERGE=""
MME_OPT_ONLY=""
OUTPUT=""
SEED=42

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --budgets)
            shift; BUDGETS=""
            while [[ $# -gt 0 && "$1" =~ ^[0-9] ]]; do
                BUDGETS="$BUDGETS $1"; shift
            done
            BUDGETS="${BUDGETS# }"
            ;;
        --datasets)
            shift; DATASETS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS="$DATASETS $1"; shift
            done
            DATASETS="${DATASETS# }"
            ;;
        --n-trials)     N_TRIALS="$2"; shift 2 ;;
        --cuda)         CUDA="$2"; shift 2 ;;
        --merge)        MERGE="--merge"; shift ;;
        --mme-opt-only) MME_OPT_ONLY="--mme-opt-only"; shift ;;
        --output)       OUTPUT="--output $2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        *) echo "[error] Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo " Pipeline: Search → Validate"
echo "========================================"
echo " Budgets:   $BUDGETS"
echo " Trials:    $N_TRIALS"
echo " Datasets:  $DATASETS"
echo " CUDA:      $CUDA"
echo " Merge:     ${MERGE:-no}"
echo " MME-opt only: ${MME_OPT_ONLY:-no}"
echo "========================================"

# ── Step 1: Search ──
echo ""
echo "[1/2] Running quantization search..."

SEARCH_CMD="$PYTHON quantization_search.py \
    --dataset mme \
    --budgets $BUDGETS \
    --n-trials $N_TRIALS \
    --cuda $CUDA \
    --seed $SEED \
    $MERGE"

echo "CMD: $SEARCH_CMD"
eval $SEARCH_CMD

# ── Find the output JSON ──
BUDGETS_TAG=$(echo $BUDGETS | tr ' ' '_')
SEARCH_JSON="results/search_mme_${BUDGETS_TAG}.json"
if [[ ! -f "$SEARCH_JSON" ]]; then
    echo "[error] Search result not found: $SEARCH_JSON"
    exit 1
fi
echo ""
echo "[search] Result saved to: $SEARCH_JSON"

# ── Step 2: Validate ──
echo ""
echo "[2/2] Running validation on: $DATASETS"

VALIDATE_CMD="$PYTHON validate_sweep.py \
    --search-results $SEARCH_JSON \
    --datasets $DATASETS \
    --cuda $CUDA \
    $MERGE \
    $MME_OPT_ONLY"

echo "CMD: $VALIDATE_CMD"
eval $VALIDATE_CMD

echo ""
echo "========================================"
echo " Done."
echo "========================================"
