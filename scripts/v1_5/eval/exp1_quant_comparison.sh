#!/bin/bash
# Experiment 1: Uniform vs Stratified Quantization under the same bit budget (266,240 bits)
# Token selection: CLS attention top-N (no merge), original llava-v1.5-7b weights
#
# Setting A: uniform B=4, N=65  → 65 × 4 × 1024 = 266,240 bits
# Setting B: uniform B=2, N=130 → 130 × 2 × 1024 = 266,240 bits
# Setting C: stratified 40×B4 + 40×B2 + 20×B1, N=100 → 266,240 bits

PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python
MODEL_PATH=/mnt/ssd/yuzhang_models/llava-v1.5-7b
CUDA=0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

export CUDA_VISIBLE_DEVICES=$CUDA
export HF_HOME=/mnt/ssd/yenhsiu_hf_cache
export LLAVA_TOKEN_METHOD=prumerge_quant

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
  "quant_mode": "$LLAVA_QUANT_MODE",
  "n_tokens": $LLAVA_N_TOKENS,
  "timestamp": "$TIMESTAMP"
}
EOF

    cd "$MME_DIR"
    $PYTHON convert_answer_to_mme.py --experiment "$EXP_NAME"
    cd eval_tool
    $PYTHON calculation.py --results_dir "answers/${EXP_NAME}" | tee "../answers/${EXP_NAME}_results.txt"
    cd "$PROJECT_ROOT"
}

# --- Setting A: uniform B=4, N=65 ---
export LLAVA_USE_QUANT=true
export LLAVA_QUANT_MODE=uniform
export LLAVA_N_TOKENS=65
export LLAVA_QUANT_BITS=4
run_mme "exp1_A_uniform_B4_N65"

# --- Setting B: uniform B=2, N=130 ---
export LLAVA_USE_QUANT=true
export LLAVA_QUANT_MODE=uniform
export LLAVA_N_TOKENS=130
export LLAVA_QUANT_BITS=2
run_mme "exp1_B_uniform_B2_N130"

# --- Setting C: stratified 40xB4 + 40xB2 + 20xB1, N=100 ---
export LLAVA_USE_QUANT=true
export LLAVA_QUANT_MODE=stratified
export LLAVA_N_TOKENS=100
export LLAVA_STRAT_GROUPS=40:4,40:2,20:1
run_mme "exp1_C_stratified_N100"

echo ""
echo "=== Experiment 1 Complete ==="
echo "Results saved in ./playground/data/eval/MME/answers/"
