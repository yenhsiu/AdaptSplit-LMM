#!/bin/bash
# Experiment 4: Budget ~115,500 bits (S=0.5Mbps), sweep top_ratio (n_4 vs n_1)
#
# top_ratio |  n_4 | n_1 |  N
# ----------|------|-----|----
#    0%     |    0 | 112 | 112  (all B=1, uniform)
#   20%     |   14 |  56 |  70
#   40%     |   20 |  31 |  51
#   60%     |   24 |  16 |  40

PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python
MODEL_PATH=/mnt/ssd/yuzhang_models/llava-v1.5-7b
CUDA=0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

export CUDA_VISIBLE_DEVICES=$CUDA
export HF_HOME=/mnt/ssd/yenhsiu_hf_cache
export LLAVA_TOKEN_METHOD=prumerge_quant
export LLAVA_USE_QUANT=true

run_mme() {
    local EXP_NAME=$1
    local MME_DIR="$PROJECT_ROOT/playground/data/eval/MME"
    local ANSWERS_FILE="$MME_DIR/answers/${EXP_NAME}.jsonl"

    echo ""
    echo "=== Running: $EXP_NAME ==="

    cd "$PROJECT_ROOT"
    $PYTHON -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --question-file "$MME_DIR/llava_mme.jsonl" \
        --image-folder "$MME_DIR/MME_Benchmark_release_version" \
        --answers-file "$ANSWERS_FILE" \
        --temperature 0 \
        --conv-mode vicuna_v1

    if [ ! -f "$ANSWERS_FILE" ]; then
        echo "ERROR: Inference failed for $EXP_NAME"
        return 1
    fi

    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    cat > "$MME_DIR/answers/${EXP_NAME}_config.json" << EOF
{
  "exp_name": "$EXP_NAME",
  "budget_bits": 115500,
  "quant_mode": "$LLAVA_QUANT_MODE",
  "n_tokens": $LLAVA_N_TOKENS,
  "strat_groups": "${LLAVA_STRAT_GROUPS:-none}",
  "quant_bits": "${LLAVA_QUANT_BITS:-none}",
  "timestamp": "$TIMESTAMP"
}
EOF

    cd "$MME_DIR"
    $PYTHON convert_answer_to_mme.py --experiment "$EXP_NAME"
    cd eval_tool
    $PYTHON calculation.py --results_dir "answers/${EXP_NAME}" | tee "../answers/${EXP_NAME}_results.txt"
    cd "$PROJECT_ROOT"
}

# top_ratio=0%: all B=1, N=112
export LLAVA_QUANT_MODE=uniform
export LLAVA_QUANT_BITS=1
export LLAVA_N_TOKENS=112
unset LLAVA_STRAT_GROUPS
run_mme "exp4_top0_B1_N112"

# top_ratio=20%: 14xB4 + 56xB1, N=70
export LLAVA_QUANT_MODE=stratified
export LLAVA_N_TOKENS=70
export LLAVA_STRAT_GROUPS=14:4,56:1
unset LLAVA_QUANT_BITS
run_mme "exp4_top20_n4x14_n1x56"

# top_ratio=40%: 20xB4 + 31xB1, N=51
export LLAVA_N_TOKENS=51
export LLAVA_STRAT_GROUPS=20:4,31:1
run_mme "exp4_top40_n4x20_n1x31"

# top_ratio=60%: 24xB4 + 16xB1, N=40
export LLAVA_N_TOKENS=40
export LLAVA_STRAT_GROUPS=24:4,16:1
run_mme "exp4_top60_n4x24_n1x16"

echo ""
echo "=== Experiment 4 Complete ==="
echo "Results saved in $PROJECT_ROOT/playground/data/eval/MME/answers/"
