#!/bin/bash XDG_CACHE_HOME='/data/shangyuzhang/'



python -m llava.eval.model_vqa_loader \
    --model-path /mnt/ssd/yenhsiu_checkpoints/llava-v1.5-7b-prunemerge-merged \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/ssd/yenhsiu_datasets/POPE/coco_val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /mnt/ssd/yenhsiu_datasets/POPE/pope_annotation \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl
