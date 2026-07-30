#!/bin/bash
# validate.sh — Cross-dataset validation with quantization config control
#
# Supported datasets: mme, textvqa, pope, scienceqa, mmbench, vqav2
#
# Usage examples:
#   bash validate.sh --dataset mme --baseline
#   bash validate.sh --dataset textvqa --no-quant --n-tokens 100
#   bash validate.sh --dataset pope --budget 266240
#   bash validate.sh --dataset mme --n4 11 --n2 5 --n1 191
#   bash validate.sh --dataset textvqa --budget 115500 --cuda 1
#   bash validate.sh --dataset scienceqa --n4 4 --n2 33 --n1 27
#   bash validate.sh --dataset mmbench --budget 266240
#   bash validate.sh --dataset vqav2 --baseline

set -e

PYTHON=python
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ── Default CUDA ──
CUDA=0

# ── Pass all args through to validate.py (extract --cuda for display only) ──
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            CUDA="$2"
            EXTRA_ARGS+=("--cuda" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=== validate.sh ==="
echo "CUDA: $CUDA"
echo "Args: ${EXTRA_ARGS[*]}"
echo "==================="

cd "$SCRIPT_DIR"
$PYTHON validate.py "${EXTRA_ARGS[@]}"
