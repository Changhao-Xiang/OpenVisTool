#!/bin/bash
# Filter distilled sessions: accuracy check + tool-call quality rules.
# Outputs an index JSONL of passing session IDs (no file copying).
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://YOUR_JUDGE_ENDPOINT/v1}"
JUDGE_MODEL="Qwen3.5-27B"

DATASET=dataset/Table/CoSyn-400K/table_difficulty_range_0.25_0.75.jsonl
SESSIONS_DIR=workspaces/qwen35plus/CoSyn-400K
OUTPUT=${SESSIONS_DIR}_filtered_index.jsonl
# OUTPUT=workspaces/qwen35plus/OS-Atlas-data/windows_filtered_index.jsonl

ACCURACY_BACKEND=judge
MATCH_MODE=generic
TOLERANCE=0.02
START_ID=385333

QUESTION_FIELD=query
ANSWER_FIELD=answer

python -m distill.filter.filter_acc_toolcall \
    --sessions-dir "$SESSIONS_DIR" \
    --dataset      "$DATASET" \
    --output       "$OUTPUT" \
    --accuracy-backend "$ACCURACY_BACKEND" \
    --tolerance    "$TOLERANCE" \
    --judge-model  "$JUDGE_MODEL" \
    --question-field "$QUESTION_FIELD" \
    --answer-field "$ANSWER_FIELD" \
    --num-workers 64 \
    --max-tool-rounds 30 \
    --match-mode "$MATCH_MODE" \
    --reject-host-paths \
    --reject-missing-images \
    --start-id "$START_ID"