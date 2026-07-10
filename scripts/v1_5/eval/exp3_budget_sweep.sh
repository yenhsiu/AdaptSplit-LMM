#!/bin/bash
# Experiment 3: Fixed budget 266,240 bits, sweep (n_4, n_2, n_1) combinations
#
# Pattern | n_4 | n_2 | n_1 |  N  | bit-tokens
# --------|-----|-----|-----|-----|------------
#    ①   |   0 |   0 | 260 | 260 | 260  (all B=1)
#    ②   |   0 | 130 |   0 | 130 | 260  (all B=2)  — reuse exp1_B
#    ③   |  65 |   0 |   0 |  65 | 260  (all B=4)  — reuse exp1_A
#    ④   |  20 |   0 | 180 | 200 | 260  (B=4 few)
#    ⑤   |  40 |  40 |  20 | 100 | 260  (stratified) — reuse exp1_C
#    ⑥   |  60 |   0 |  20 |  80 | 260  (B=4 heavy)
#    ⑦   |   0 |  90 |  80 | 170 | 260  (B=2 heavy)

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
  "method": "prumerge_quant",
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

# ① all B=1, N=260
export LLAVA_QUANT_MODE=uniform
export LLAVA_N_TOKENS=260
export LLAVA_QUANT_BITS=1
unset LLAVA_STRAT_GROUPS
run_mme "exp3_1_B1_N260"

# ④ stratified 20xB4 + 180xB1, N=200
export LLAVA_QUANT_MODE=stratified
export LLAVA_N_TOKENS=200
export LLAVA_STRAT_GROUPS=20:4,180:1
unset LLAVA_QUANT_BITS
run_mme "exp3_4_strat_n4x20_n1x180"

# ⑥ stratified 60xB4 + 20xB1, N=80
export LLAVA_N_TOKENS=80
export LLAVA_STRAT_GROUPS=60:4,20:1
run_mme "exp3_6_strat_n4x60_n1x20"

# ⑦ stratified 90xB2 + 80xB1, N=170
export LLAVA_N_TOKENS=170
export LLAVA_STRAT_GROUPS=90:2,80:1
run_mme "exp3_7_strat_n2x90_n1x80"

echo ""
echo "=== Experiment 3 Complete ==="
echo "Reuse from Exp1 — ②: exp1_B_uniform_B2_N130  ③: exp1_A_uniform_B4_N65  ⑤: exp1_C_stratified_N100"
echo "Results saved in $PROJECT_ROOT/playground/data/eval/MME/answers/"
