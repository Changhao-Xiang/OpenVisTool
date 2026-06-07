#!/bin/bash
# Compute avg@k for the base model when given the teacher's tool_response prefix.
# Only computes the metric and writes per-item JSONL — no filtering happens here.
export OPENAI_API_KEY="EMPTY"

ROLLOUT_API_BASE="${ROLLOUT_API_BASE:-https://YOUR_ROLLOUT_ENDPOINT/v1}"
ROLLOUT_MODEL="Qwen3.5-9B"
JUDGE_API_BASE="${JUDGE_API_BASE:-https://YOUR_JUDGE_ENDPOINT/v1}"
JUDGE_MODEL="Qwen3.5-27B"

DATASET=dataset/Code/VinciCoder-1.6M-SFT/vincicoder_bucket_selected_difficulty_filtered_qwen35_9b.jsonl
MEDIA_DIR=dataset/Code/VinciCoder-1.6M-SFT/images
SESSIONS_DIR=workspaces/qwen35plus/VinciCoder
SAMPLE_RANGE=0:15000
OUTPUT=${DATASET%.jsonl}_tool_gain_prefix.jsonl
# Optional: same prompt md you used in difficulty filtering (leave empty if none).
SYSTEM_PROMPT_MD=""

ACCURACY_BACKEND=html_vlm
MATCH_MODE=point
TOLERANCE=0.02

K=4
TEMPERATURE=0.7
MAX_TOKENS=16384
PER_RESPONSE_CHAR_LIMIT=10000
NUM_WORKERS=32
SAMPLE_SUFFIX=""   # set to "_s0" if teacher was run with --sample-index 0

python distill/filter/filter_tool_gain_with_prefix.py \
    --dataset "$DATASET" \
    --media-dir "$MEDIA_DIR" \
    --sessions-dir "$SESSIONS_DIR" \
    --sample-suffix "$SAMPLE_SUFFIX" \
    --output "$OUTPUT" \
    --model "$ROLLOUT_MODEL" \
    --rollout-api-base "$ROLLOUT_API_BASE" \
    --k "$K" \
    --accuracy-backend "$ACCURACY_BACKEND" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-base "$JUDGE_API_BASE" \
    --tolerance "$TOLERANCE" \
    --match-mode "$MATCH_MODE" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --per-response-char-limit "$PER_RESPONSE_CHAR_LIMIT" \
    --num-workers "$NUM_WORKERS" \
    --sample-range "$SAMPLE_RANGE" \
    ${SYSTEM_PROMPT_MD:+--system-prompt-md "$SYSTEM_PROMPT_MD"}
