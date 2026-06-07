#!/bin/bash
# Filter distilled HTML-generation sessions: VLM visual-consistency scoring +
# tool-call quality rules. Outputs an index JSONL of passing session IDs.
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://YOUR_JUDGE_ENDPOINT/v1}"
JUDGE_MODEL="Qwen3.5-27B"

DATASET=dataset/Code/VinciCoder-1.6M-SFT/vincicoder_bucket_selected_difficulty_filtered_qwen35_9b.jsonl
SESSIONS_DIR=workspaces/qwen35plus/VinciCoder
OUTPUT=${SESSIONS_DIR}_filtered_index.jsonl

ACCURACY_BACKEND=html_vlm
HTML_SCORE_THRESHOLD=80.0

QUESTION_FIELD=query
ANSWER_FIELD=answer
MEDIA_FIELD=images

python -m distill.filter.filter_acc_toolcall \
    --sessions-dir "$SESSIONS_DIR" \
    --dataset      "$DATASET" \
    --output       "$OUTPUT" \
    --accuracy-backend "$ACCURACY_BACKEND" \
    --judge-model  "$JUDGE_MODEL" \
    --question-field "$QUESTION_FIELD" \
    --answer-field "$ANSWER_FIELD" \
    --media-field  "$MEDIA_FIELD" \
    --html-score-threshold "$HTML_SCORE_THRESHOLD" \
    --num-workers 32 \
    --max-tool-rounds 30 \
    --reject-host-paths \
    --reject-missing-images
