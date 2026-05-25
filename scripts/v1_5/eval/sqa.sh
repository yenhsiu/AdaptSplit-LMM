#!/bin/bash
PYTHON=/mnt/ssd/yenhsiu_envs/llava_eval/bin/python

# ============================================================
# Configuration — only change this section
# ============================================================
CUDA=0
METHOD=original          # original | prumerge | prumerge_plus
USE_QUANT=false          # true | false
QUANT_BITS=4             # 2 | 4 | 8
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

echo "=== SQA Evaluation ==="
echo "Method:     $METHOD"
echo "Quant:      $USE_QUANT (${QUANT_BITS}bit)"
echo "CUDA:       $CUDA"
echo "Exp name:   $EXP_NAME"
echo "Model path: $MODEL_PATH"
echo "======================"

ANSWERS_FILE=./playground/data/eval/scienceqa/answers/${EXP_NAME}.jsonl

if [ -n "$MODEL_BASE" ]; then
    $PYTHON -m llava.eval.model_vqa_science \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file "$ANSWERS_FILE" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1
else
    $PYTHON -m llava.eval.model_vqa_science \
        --model-path "$MODEL_PATH" \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
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
CONFIG_FILE="./playground/data/eval/scienceqa/answers/${EXP_NAME}_config.json"
cat > "$CONFIG_FILE" << EOF
{
  "exp_name": "$EXP_NAME",
  "method": "$METHOD",
  "use_quant": $USE_QUANT,
  "quant_bits": $QUANT_BITS,
  "cuda": $CUDA,
  "model_path": "$MODEL_PATH",
  "timestamp": "$TIMESTAMP"
}
EOF

$PYTHON llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file "$ANSWERS_FILE" \
    --output-file "./playground/data/eval/scienceqa/answers/${EXP_NAME}_output.jsonl" \
    --output-result "./playground/data/eval/scienceqa/answers/${EXP_NAME}_result.json" \
    | tee "./playground/data/eval/scienceqa/answers/${EXP_NAME}_results.txt"
