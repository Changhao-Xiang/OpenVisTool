#!/bin/bash
# Select samples with positive tool-call gain by comparing notool avg@k against
# with-prefix avg@k. Inputs are produced by filter_difficulty.py and
# filter_tool_gain_with_prefix.py respectively.

DATASET=dataset/VisualSearch/Vero-600k-visual-search_difficulty_filtered_qwen35_9b.jsonl
WITH_PREFIX=${DATASET%.jsonl}_tool_gain_prefix.jsonl
OUTPUT=${DATASET%_difficulty_filtered_qwen35_9b.jsonl}_tool_gain_selected.jsonl
REPORT=${OUTPUT%.jsonl}_report.json

GAIN_THRESHOLD=0.1

python distill/select/select_tool_gain.py \
    --notool-jsonl "$DATASET" \
    --with-prefix-jsonl "$WITH_PREFIX" \
    --output "$OUTPUT" \
    --report "$REPORT" \
    --gain-threshold "$GAIN_THRESHOLD"
