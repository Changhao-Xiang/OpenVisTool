#!/bin/bash
# "Correctness + tool-gain" filter for a single dataset: intersect a
# correctness-filtered index with a tool-gain-selected index.
#
# Edit the three paths below, then run.

CORRECTNESS_INDEX=workspaces/qwen35plus/ChartVerse-SFT-filtered_index.jsonl
TOOL_GAIN_INDEX=dataset/OpenVisTool_toolgain_only/tool-gain-report/Chart__ChartVerse-SFT-100K_difficulty_filtered_qwen35_9b_tool_gain_selected.jsonl
OUTPUT=${TOOL_GAIN_INDEX%_tool_gain_selected.jsonl}_correct_and_gain.jsonl
REPORT=${OUTPUT%.jsonl}_report.json

python distill/select/select_correct_and_gain.py \
    --correctness-index "$CORRECTNESS_INDEX" \
    --tool-gain-index "$TOOL_GAIN_INDEX" \
    --output "$OUTPUT" \
    --report "$REPORT"
