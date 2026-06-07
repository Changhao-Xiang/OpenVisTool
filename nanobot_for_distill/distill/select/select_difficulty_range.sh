#!/bin/bash
# Select samples whose avg@k falls within [min_avg, max_avg].

DATASET=dataset/Table/CoSyn-400K/table.jsonl
PROGRESS=${DATASET%.jsonl}_difficulty_filtered_qwen35_9b.progress.jsonl

MIN_AVG=0.25
MAX_AVG=0.75

# OUTPUT defaults to <dataset_stem>_difficulty_range_<min>_<max>.jsonl if unset
# OUTPUT=${DATASET%.jsonl}_difficulty_range_${MIN_AVG}_${MAX_AVG}.jsonl

python distill/select/select_difficulty_range.py \
    --dataset "$DATASET" \
    --progress "$PROGRESS" \
    --min-avg "$MIN_AVG" \
    --max-avg "$MAX_AVG"
    # --output "$OUTPUT"
