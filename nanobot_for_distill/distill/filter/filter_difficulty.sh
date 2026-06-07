#!/bin/bash
# Filter hard samples using no-tool vLLM rollouts.
export OPENAI_API_KEY="EMPTY"
ROLLOUT_API_BASE="${ROLLOUT_API_BASE:-https://YOUR_ROLLOUT_ENDPOINT/v1}"
ROLLOUT_MODEL="Qwen3.5-9B"
JUDGE_API_BASE="${JUDGE_API_BASE:-https://YOUR_JUDGE_ENDPOINT/v1}"
JUDGE_MODEL="Qwen3.5-27B"

DATASET=dataset/GUI/OS-Atlas-data/os_atlas_macos_highres_small_bbox.jsonl
MEDIA_DIR=dataset/GUI/OS-Atlas-data/images_macos
OUTPUT=${DATASET%.jsonl}_difficulty_filtered_qwen35_9b.jsonl

# SYSTEM_PROMPT_MD=distill/filter/rollout_prompts/default.md
SYSTEM_PROMPT_MD=distill/filter/rollout_prompts/gui_grounding_qwen35.md
ACCURACY_BACKEND=rule
MATCH_MODE=bbox

SAMPLE_RANGE=0:
NUM_WORKERS=80

K=4
TOLERANCE=0.01
THRESHOLD=0.6
TEMPERATURE=0.7
MAX_TOKENS=32768

python distill/filter/filter_difficulty.py \
    --dataset "$DATASET" \
    --media-dir "$MEDIA_DIR" \
    --output "$OUTPUT" \
    --model "$ROLLOUT_MODEL" \
    --rollout-api-base "$ROLLOUT_API_BASE" \
    --k "$K" \
    --accuracy-backend "$ACCURACY_BACKEND" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-base "$JUDGE_API_BASE" \
    --tolerance "$TOLERANCE" \
    --threshold "$THRESHOLD" \
    --num-workers "$NUM_WORKERS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --sample-range "$SAMPLE_RANGE" \
    --match-mode "$MATCH_MODE" \
    --system-prompt-md "$SYSTEM_PROMPT_MD"
