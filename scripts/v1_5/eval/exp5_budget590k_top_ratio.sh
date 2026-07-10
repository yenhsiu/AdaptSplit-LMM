#!/bin/bash
# Experiment 5: Budget 589,824 bits (S=2.5Mbps), sweep top_ratio (n_4 vs n_1)
#
# top_ratio |  n_4 | n_1 |   N
# ----------|------|-----|-----
#    0%     |    0 | 576 | 576  (all B=1, uniform)
#   20%     |   72 | 288 | 360
#   40%     |  105 | 157 | 262
#   60%     |  123 |  83 | 206

PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python
MODEL_PATH=/mnt/ssd/yuzhang_models/llava-v1.5-7b
CUDA=1
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
  "budget_bits": 589824,
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

# top_ratio=0%: all B=1, N=576
export LLAVA_QUANT_MODE=uniform
export LLAVA_QUANT_BITS=1
export LLAVA_N_TOKENS=576
unset LLAVA_STRAT_GROUPS
run_mme "exp5_top0_B1_N576"

# top_ratio=20%: 72xB4 + 288xB1, N=360
export LLAVA_QUANT_MODE=stratified
export LLAVA_N_TOKENS=360
export LLAVA_STRAT_GROUPS=72:4,288:1
unset LLAVA_QUANT_BITS
run_mme "exp5_top20_n4x72_n1x288"

# top_ratio=40%: 105xB4 + 157xB1, N=262
export LLAVA_N_TOKENS=262
export LLAVA_STRAT_GROUPS=105:4,157:1
run_mme "exp5_top40_n4x105_n1x157"

# top_ratio=60%: 123xB4 + 83xB1, N=206
export LLAVA_N_TOKENS=206
export LLAVA_STRAT_GROUPS=123:4,83:1
run_mme "exp5_top60_n4x123_n1x83"

echo ""
echo "=== Experiment 5 Complete ==="
echo "Results saved in $PROJECT_ROOT/playground/data/eval/MME/answers/"
