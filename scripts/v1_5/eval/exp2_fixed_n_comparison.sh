#!/bin/bash
# Experiment 2: Fixed N=100, vary bits/token to observe accuracy vs transmission budget
#
# Setting D: N=100, no quant  (FP16, 1,638,400 bits) — upper bound
# Setting E: N=100, B=4       (409,600 bits)
# Setting C: N=100, stratified 40xB4+40xB2+20xB1 (266,240 bits) — reuse Exp1 results
# Setting F: N=100, B=2       (204,800 bits)
# Setting G: N=100, B=1       (102,400 bits)

PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python
MODEL_PATH=/mnt/ssd/yuzhang_models/llava-v1.5-7b
CUDA=0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

export CUDA_VISIBLE_DEVICES=$CUDA
export HF_HOME=/mnt/ssd/yenhsiu_hf_cache
export LLAVA_TOKEN_METHOD=prumerge_quant
export LLAVA_N_TOKENS=50

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
  "method": "prumerge_quant",
  "use_quant": $LLAVA_USE_QUANT,
  "quant_mode": "${LLAVA_QUANT_MODE:-none}",
  "n_tokens": $LLAVA_N_TOKENS,
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

# --- Setting D: N=100, no quantization (upper bound) ---
export LLAVA_USE_QUANT=false
unset LLAVA_QUANT_MODE
unset LLAVA_QUANT_BITS
run_mme "exp2_D_noq_N100"

# --- Setting E: N=100, uniform B=4 ---
export LLAVA_USE_QUANT=true
export LLAVA_QUANT_MODE=uniform
export LLAVA_QUANT_BITS=4
run_mme "exp2_E_uniform_B4_N50"

# --- Setting F: N=100, uniform B=2 ---
export LLAVA_QUANT_BITS=2
run_mme "exp2_F_uniform_B2_N50"

# --- Setting G: N=100, uniform B=1 ---
export LLAVA_QUANT_BITS=1
run_mme "exp2_G_uniform_B1_N50"

echo ""
echo "=== Experiment 2 Complete ==="
echo "Setting C (stratified, N=100) results already available from Exp1: exp1_C_stratified_N50"
echo "Results saved in $PROJECT_ROOT/playground/data/eval/MME/answers/"
