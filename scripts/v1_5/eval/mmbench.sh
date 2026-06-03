#!/bin/bash
PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python

# ============================================================
# Configuration — only change this section
# ============================================================
CUDA=0
METHOD=prumerge_plus          # original | prumerge | prumerge_plus
USE_QUANT=true          # true | false
QUANT_BITS=4             # 1 | 2 | 4
SPLIT=mmbench_dev_20230712  # mmbench_dev_20230712 | mmbench_test_20230712
# ============================================================

# Auto model path mapping
if [ "$METHOD" = "original" ]; then
    MODEL_PATH=/mnt/ssd/yuzhang_models/llava-v1.5-7b
    MODEL_BASE=""
elif [ "$METHOD" = "prumerge" ]; then
    MODEL_PATH=/mnt/ssd/yuzhang_models/llava-prumerge-vicuna-7b-v1.5-lora
    MODEL_BASE=lmsys/vicuna-7b-v1.5
elif [ "$METHOD" = "prumerge_plus" ]; then
    MODEL_PATH=/mnt/ssd/yuzhang_models/llava-prumerge-plus-vicuna-7b-v1.5-lora
    MODEL_BASE=lmsys/vicuna-7b-v1.5
else
    echo "Unknown METHOD: $METHOD. Use original | prumerge | prumerge_plus"
    exit 1
fi

# Auto experiment name
EXP_NAME="$METHOD"
if [ "$USE_QUANT" = "true" ]; then
    EXP_NAME="${EXP_NAME}_quant${QUANT_BITS}bit"
fi

# Export env vars for Python
export CUDA_VISIBLE_DEVICES=$CUDA
export HF_HOME=/mnt/ssd/yenhsiu_hf_cache
export LLAVA_TOKEN_METHOD=$METHOD
export LLAVA_USE_QUANT=$USE_QUANT
export LLAVA_QUANT_BITS=$QUANT_BITS

echo "=== MMBench Evaluation ==="
echo "Method:     $METHOD"
echo "Quant:      $USE_QUANT (${QUANT_BITS}bit)"
echo "CUDA:       $CUDA"
echo "Split:      $SPLIT"
echo "Exp name:   $EXP_NAME"
echo "Model path: $MODEL_PATH"
echo "=========================="

ANSWERS_DIR=./playground/data/eval/mmbench/answers/$SPLIT
ANSWERS_FILE=$ANSWERS_DIR/${EXP_NAME}.jsonl

mkdir -p "$ANSWERS_DIR"

if [ -n "$MODEL_BASE" ]; then
    $PYTHON -m llava.eval.model_vqa_mmbench \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file ./playground/data/eval/mmbench/${SPLIT}.tsv \
        --answers-file "$ANSWERS_FILE" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1
else
    $PYTHON -m llava.eval.model_vqa_mmbench \
        --model-path "$MODEL_PATH" \
        --question-file ./playground/data/eval/mmbench/${SPLIT}.tsv \
        --answers-file "$ANSWERS_FILE" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1
fi

if [ ! -f "$ANSWERS_FILE" ]; then
    echo "ERROR: Inference failed, answers file not found: $ANSWERS_FILE"
    exit 1
fi

# Save config
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
CONFIG_FILE="$ANSWERS_DIR/${EXP_NAME}_config.json"
cat > "$CONFIG_FILE" << EOF
{
  "exp_name": "$EXP_NAME",
  "method": "$METHOD",
  "use_quant": $USE_QUANT,
  "quant_bits": $QUANT_BITS,
  "cuda": $CUDA,
  "split": "$SPLIT",
  "model_path": "$MODEL_PATH",
  "timestamp": "$TIMESTAMP"
}
EOF

# Convert for submission (upload to https://mmbench.opencompass.org.cn)
UPLOAD_DIR=./playground/data/eval/mmbench/answers_upload/$SPLIT
mkdir -p "$UPLOAD_DIR"

# Local scoring (dev split only — has ground truth answers)
RESULTS_TXT="$ANSWERS_DIR/${EXP_NAME}_results.txt"
$PYTHON scripts/eval_mmbench_local.py \
    --annotation-file ./playground/data/eval/mmbench/${SPLIT}.tsv \
    --result-file "$ANSWERS_FILE" \
    | tee "$RESULTS_TXT"

# Convert for submission (upload to https://mmbench.opencompass.org.cn)
$PYTHON scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/${SPLIT}.tsv \
    --result-dir "$ANSWERS_DIR" \
    --upload-dir "$UPLOAD_DIR" \
    --experiment "$EXP_NAME"

echo ""
echo "Submission file saved to: $UPLOAD_DIR/${EXP_NAME}.xlsx"
echo "Upload it at: https://mmbench.opencompass.org.cn/mmbench-submission"
